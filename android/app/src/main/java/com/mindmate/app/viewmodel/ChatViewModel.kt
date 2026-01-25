package com.mindmate.app.viewmodel

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mindmate.app.llm.LlamaInference
import com.mindmate.app.model.ChatState
import com.mindmate.app.model.ChatUiState
import com.mindmate.app.model.Message
import com.mindmate.app.model.MessageRole
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    
    companion object {
        private const val TAG = "ChatViewModel"
    }
    
    private val llamaInference = LlamaInference(application)
    
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()
    
    fun selectModel() {
        _uiState.update { it.copy(chatState = ChatState.SelectingModel) }
    }
    
    fun loadModel(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(chatState = ChatState.LoadingModel(0f)) }
            
            try {
                // Get the actual file path from URI
                val modelPath = getPathFromUri(uri)
                if (modelPath == null) {
                    _uiState.update { 
                        it.copy(chatState = ChatState.Error("Could not access model file"))
                    }
                    return@launch
                }
                
                val modelName = File(modelPath).name
                
                llamaInference.loadModel(modelPath) { progress ->
                    _uiState.update { 
                        it.copy(chatState = ChatState.LoadingModel(progress))
                    }
                }.onSuccess {
                    _uiState.update { 
                        it.copy(
                            chatState = ChatState.Ready,
                            modelPath = modelPath,
                            modelName = modelName
                        )
                    }
                    Log.d(TAG, "Model loaded: $modelName")
                }.onFailure { error ->
                    Log.e(TAG, "Failed to load model", error)
                    _uiState.update { 
                        it.copy(chatState = ChatState.Error(
                            error.message ?: "Failed to load model"
                        ))
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                _uiState.update { 
                    it.copy(chatState = ChatState.Error(e.message ?: "Unknown error"))
                }
            }
        }
    }
    
    private fun getPathFromUri(uri: Uri): String? {
        val context = getApplication<Application>()
        
        // For content:// URIs, we need to copy the file to app's cache
        return try {
            if (uri.scheme == "file") {
                uri.path
            } else {
                // Copy to cache for content:// URIs
                val inputStream = context.contentResolver.openInputStream(uri)
                val cacheFile = File(context.cacheDir, "model_${System.currentTimeMillis()}.gguf")
                
                inputStream?.use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                
                cacheFile.absolutePath
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting path from URI", e)
            null
        }
    }
    
    fun sendMessage(content: String) {
        if (content.isBlank()) return
        if (_uiState.value.chatState != ChatState.Ready) return
        
        val userMessage = Message(
            role = MessageRole.USER,
            content = content
        )
        
        _uiState.update { state ->
            state.copy(
                messages = state.messages + userMessage,
                chatState = ChatState.Generating
            )
        }
        
        // Create placeholder for assistant response
        val assistantMessage = Message(
            role = MessageRole.ASSISTANT,
            content = "",
            isStreaming = true
        )
        
        _uiState.update { state ->
            state.copy(messages = state.messages + assistantMessage)
        }
        
        viewModelScope.launch {
            val conversationHistory = buildConversationHistory()
            var fullResponse = ""
            
            llamaInference.generateResponse(content, conversationHistory)
                .catch { error ->
                    Log.e(TAG, "Generation error", error)
                    _uiState.update { state ->
                        val updatedMessages = state.messages.dropLast(1) + Message(
                            id = assistantMessage.id,
                            role = MessageRole.ASSISTANT,
                            content = "I'm sorry, I encountered an error. Please try again.",
                            isStreaming = false
                        )
                        state.copy(
                            messages = updatedMessages,
                            chatState = ChatState.Ready
                        )
                    }
                }
                .collect { token ->
                    fullResponse += token
                    _uiState.update { state ->
                        val updatedMessages = state.messages.dropLast(1) + Message(
                            id = assistantMessage.id,
                            role = MessageRole.ASSISTANT,
                            content = fullResponse,
                            isStreaming = true
                        )
                        state.copy(messages = updatedMessages)
                    }
                }
            
            // Mark generation as complete
            _uiState.update { state ->
                val updatedMessages = state.messages.dropLast(1) + Message(
                    id = assistantMessage.id,
                    role = MessageRole.ASSISTANT,
                    content = fullResponse.trim(),
                    isStreaming = false
                )
                state.copy(
                    messages = updatedMessages,
                    chatState = ChatState.Ready
                )
            }
        }
    }
    
    private fun buildConversationHistory(): List<Pair<String, String>> {
        return _uiState.value.messages
            .filter { it.role != MessageRole.SYSTEM && !it.isStreaming }
            .dropLast(1) // Exclude the message we just added
            .map { message ->
                val role = when (message.role) {
                    MessageRole.USER -> "user"
                    MessageRole.ASSISTANT -> "assistant"
                    MessageRole.SYSTEM -> "system"
                }
                role to message.content
            }
    }
    
    fun clearError() {
        if (_uiState.value.chatState is ChatState.Error) {
            _uiState.update { 
                it.copy(chatState = if (llamaInference.isModelLoaded()) {
                    ChatState.Ready
                } else {
                    ChatState.Initial
                })
            }
        }
    }
    
    fun clearChat() {
        _uiState.update { it.copy(messages = emptyList()) }
    }
    
    override fun onCleared() {
        super.onCleared()
        llamaInference.unloadModel()
    }
}
