package com.mindmate.app.llm

import android.content.Context
import android.util.Log
import com.mindmate.llama.Llm
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

class LlamaInference(private val context: Context) {
    
    companion object {
        private const val TAG = "LlamaInference"
        private const val MAX_TOKENS = 512
    }
    
    private val llm = Llm()
    private var systemPrompt: String = ""
    
    init {
        loadSystemPrompt()
    }
    
    private fun loadSystemPrompt() {
        try {
            context.resources.openRawResource(
                context.resources.getIdentifier("system_prompt", "raw", context.packageName)
            ).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    systemPrompt = reader.readText()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load system prompt from resources, using default")
            systemPrompt = """You are MindMate, a warm, emotionally intelligent companion.
                |Be conversational, empathetic, and supportive.
                |Help the user feel understood and heard in a real, human way.""".trimMargin()
        }
    }
    
    suspend fun loadModel(
        modelPath: String,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Loading model from: $modelPath")
            onProgress(0.1f)
            
            onProgress(0.3f)
            
            val success = llm.load(modelPath)
            
            onProgress(1.0f)
            
            if (success) {
                Log.d(TAG, "Model loaded successfully")
                Result.success(Unit)
            } else {
                Log.e(TAG, "Failed to load model")
                Result.failure(RuntimeException("Failed to load model"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            Result.failure(e)
        }
    }
    
    fun isModelLoaded(): Boolean = llm.isLoaded()
    
    fun generateResponse(
        userMessage: String,
        conversationHistory: List<Pair<String, String>> = emptyList()
    ): Flow<String> = flow {
        if (!llm.isLoaded()) {
            throw IllegalStateException("Model not loaded")
        }
        
        // Build the full prompt with conversation history
        val prompt = buildPrompt(userMessage, conversationHistory)
        
        Log.d(TAG, "Generating response for prompt length: ${prompt.length}")
        
        try {
            // Generate full response (native lib doesn't support streaming yet)
            val response = llm.generate(prompt, MAX_TOKENS)
            
            // Emit the full response
            if (response.isNotEmpty()) {
                emit(response)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation", e)
            throw e
        }
    }.flowOn(Dispatchers.IO)
    
    private fun buildPrompt(
        userMessage: String,
        conversationHistory: List<Pair<String, String>>
    ): String {
        val sb = StringBuilder()
        
        // Llama 3.2 chat format
        sb.append("<|begin_of_text|>")
        sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
        sb.append(systemPrompt)
        sb.append("<|eot_id|>")
        
        // Add conversation history
        for ((role, content) in conversationHistory) {
            sb.append("<|start_header_id|>$role<|end_header_id|>\n\n")
            sb.append(content)
            sb.append("<|eot_id|>")
        }
        
        // Add current user message
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
        sb.append(userMessage)
        sb.append("<|eot_id|>")
        
        // Start assistant response
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return sb.toString()
    }
    
    fun unloadModel() {
        try {
            llm.unload()
            Log.d(TAG, "Model unloaded")
        } catch (e: Exception) {
            Log.e(TAG, "Error unloading model", e)
        }
    }
}
