package com.mindmate.app.model

sealed class ChatState {
    object Initial : ChatState()
    object SelectingModel : ChatState()
    data class LoadingModel(val progress: Float = 0f) : ChatState()
    object Ready : ChatState()
    object Generating : ChatState()
    data class Error(val message: String) : ChatState()
}

data class ChatUiState(
    val chatState: ChatState = ChatState.Initial,
    val messages: List<Message> = emptyList(),
    val modelPath: String? = null,
    val modelName: String? = null
)
