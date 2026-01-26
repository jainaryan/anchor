package com.mindmate.llama

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Kotlin wrapper for llama.cpp native library.
 * Provides model loading and text generation capabilities.
 */
class Llm {
    
    companion object {
        private const val TAG = "Llm"
        
        init {
            try {
                System.loadLibrary("llama-android")
                Log.d(TAG, "Native library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            }
        }
    }
    
    private var modelPtr: Long = 0
    
    /**
     * Load a GGUF model from the given path
     */
    suspend fun load(modelPath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Loading model: $modelPath")
            modelPtr = loadModel(modelPath)
            val success = modelPtr != 0L
            Log.d(TAG, "Model load result: $success")
            success
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            false
        }
    }
    
    /**
     * Generate text based on the given prompt
     */
    suspend fun generate(prompt: String, maxTokens: Int = 512): String = withContext(Dispatchers.IO) {
        if (!isLoaded()) {
            Log.w(TAG, "Model not loaded")
            return@withContext ""
        }
        
        try {
            generateNative(prompt, maxTokens)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating", e)
            ""
        }
    }
    
    /**
     * Check if model is loaded
     */
    fun isLoaded(): Boolean = isModelLoaded()
    
    /**
     * Unload the model and free resources
     */
    fun unload() {
        if (modelPtr != 0L) {
            unloadModel()
            modelPtr = 0
            Log.d(TAG, "Model unloaded")
        }
    }
    
    // Native methods
    private external fun loadModel(modelPath: String): Long
    private external fun unloadModel()
    private external fun generateNative(prompt: String, maxTokens: Int): String
    private external fun isModelLoaded(): Boolean
}
