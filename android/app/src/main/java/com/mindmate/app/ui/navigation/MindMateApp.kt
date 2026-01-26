package com.mindmate.app.ui.navigation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
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
import com.mindmate.app.viewmodel.ChatViewModel

@Composable
fun MindMateApp(
    viewModel: ChatViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Always show ChatScreen as the base
        ChatScreen(
            uiState = uiState,
            onSendMessage = { message ->
                viewModel.sendMessage(message)
            },
            onClearChat = {
                viewModel.clearChat()
            }
        )
        
        // Show loading overlay when loading model
        when (val state = uiState.chatState) {
            is ChatState.Initial, is ChatState.LoadingModel -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(MaterialTheme.colorScheme.background.copy(alpha = 0.95f)),
                    contentAlignment = Alignment.Center
                ) {
                    val progress = if (state is ChatState.LoadingModel) state.progress else 0f
                    LoadingIndicator(
                        message = "Loading MindMate...\nThis may take a minute",
                        progress = progress
                    )
                }
            }
            
            is ChatState.Error -> {
                AlertDialog(
                    onDismissRequest = { viewModel.retryLoadModel() },
                    title = { Text("Loading Failed") },
                    text = { Text(state.message) },
                    confirmButton = {
                        TextButton(onClick = { viewModel.retryLoadModel() }) {
                            Text("Retry")
                        }
                    }
                )
            }
            
            else -> { /* ChatScreen is already shown */ }
        }
    }
}

