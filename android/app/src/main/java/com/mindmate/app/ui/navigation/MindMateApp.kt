package com.mindmate.app.ui.navigation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mindmate.app.model.ChatState
import com.mindmate.app.ui.components.LoadingIndicator
import com.mindmate.app.ui.screens.ChatScreen
import com.mindmate.app.ui.screens.ModelLoadScreen
import com.mindmate.app.viewmodel.ChatViewModel

@Composable
fun MindMateApp(
    viewModel: ChatViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Box(modifier = Modifier.fillMaxSize()) {
        when (val state = uiState.chatState) {
            is ChatState.Initial, is ChatState.SelectingModel -> {
                ModelLoadScreen(
                    onModelSelected = { uri ->
                        viewModel.loadModel(uri)
                    }
                )
            }
            
            is ChatState.LoadingModel -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    LoadingIndicator(
                        message = "Loading MindMate model...\nThis may take a minute",
                        progress = state.progress
                    )
                }
            }
            
            is ChatState.Ready, is ChatState.Generating -> {
                ChatScreen(
                    uiState = uiState,
                    onSendMessage = { message ->
                        viewModel.sendMessage(message)
                    },
                    onClearChat = {
                        viewModel.clearChat()
                    }
                )
            }
            
            is ChatState.Error -> {
                // Show error dialog over the appropriate screen
                if (uiState.modelPath == null) {
                    ModelLoadScreen(
                        onModelSelected = { uri ->
                            viewModel.loadModel(uri)
                        }
                    )
                } else {
                    ChatScreen(
                        uiState = uiState,
                        onSendMessage = { },
                        onClearChat = { }
                    )
                }
                
                AlertDialog(
                    onDismissRequest = { viewModel.clearError() },
                    title = { Text("Error") },
                    text = { Text(state.message) },
                    confirmButton = {
                        TextButton(onClick = { viewModel.clearError() }) {
                            Text("OK")
                        }
                    }
                )
            }
        }
    }
}
