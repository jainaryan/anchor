#include <android/log.h>
#include <jni.h>
#include <string>
#include <thread>

#include "common.h"
#include "llama.h"

#define TAG "llama-android"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Global state
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static llama_sampler *g_sampler = nullptr;

// Log callback
static void log_callback(ggml_log_level level, const char *text,
                         void *user_data) {
  if (level == GGML_LOG_LEVEL_ERROR) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", text);
  } else if (level == GGML_LOG_LEVEL_WARN) {
    __android_log_print(ANDROID_LOG_WARN, TAG, "%s", text);
  } else {
    __android_log_print(ANDROID_LOG_INFO, TAG, "%s", text);
  }
}

extern "C" {

JNIEXPORT jlong JNICALL Java_com_mindmate_llama_Llm_loadModel(
    JNIEnv *env, jobject thiz, jstring model_path) {
  const char *path = env->GetStringUTFChars(model_path, nullptr);
  LOGi("Loading model from: %s", path);

  // Initialize llama backend and logging
  llama_log_set(log_callback, nullptr);
  llama_backend_init();

  // Model parameters
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // CPU only for Android
  model_params.use_mmap = true; // Enable mmap for better memory usage on device

  // Load model
  g_model = llama_model_load_from_file(path, model_params);
  env->ReleaseStringUTFChars(model_path, path);

  if (g_model == nullptr) {
    LOGe("Failed to load model");
    return 0;
  }

  // Context parameters
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512; // Minimal context to rule out OOM
  ctx_params.n_batch = 8; // Minimal batch to rule out compute OOM
  ctx_params.n_threads = 4;
  ctx_params.n_threads_batch = 4;

  // Create context
  g_ctx = llama_init_from_model(g_model, ctx_params);
  if (g_ctx == nullptr) {
    LOGe("Failed to create context");
    llama_model_free(g_model);
    g_model = nullptr;
    return 0;
  }

  // Create sampler
  llama_sampler_chain_params sampler_params =
      llama_sampler_chain_default_params();
  g_sampler = llama_sampler_chain_init(sampler_params);
  llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(g_sampler,
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  LOGi("Model loaded successfully");
  return reinterpret_cast<jlong>(g_model);
}

JNIEXPORT void JNICALL Java_com_mindmate_llama_Llm_unloadModel(JNIEnv *env,
                                                               jobject thiz) {
  if (g_sampler != nullptr) {
    llama_sampler_free(g_sampler);
    g_sampler = nullptr;
  }
  if (g_ctx != nullptr) {
    llama_free(g_ctx);
    g_ctx = nullptr;
  }
  if (g_model != nullptr) {
    llama_model_free(g_model);
    g_model = nullptr;
  }
  llama_backend_free();
  LOGi("Model unloaded");
}

JNIEXPORT jstring JNICALL Java_com_mindmate_llama_Llm_generateNative(
    JNIEnv *env, jobject thiz, jstring prompt, jint max_tokens) {
  if (g_model == nullptr || g_ctx == nullptr || g_sampler == nullptr) {
    LOGe("Model not fully initialized");
    return env->NewStringUTF("Error: Model not initialized");
  }

  try {
    const char *prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str(prompt_cstr);
    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    LOGi("Generating with prompt length: %zu", prompt_str.length());

    // Tokenize
    LOGi("Starting tokenization");
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    if (vocab == nullptr) {
      LOGe("Vocab is null");
      return env->NewStringUTF("Error: Vocab is null");
    }

    std::vector<llama_token> tokens =
        common_tokenize(vocab, prompt_str, true, true);
    LOGi("Tokenization complete, %zu tokens", tokens.size());

    int n_ctx = llama_n_ctx(g_ctx);
    if ((int)tokens.size() > n_ctx - 4) {
      LOGe("Prompt too long");
      return env->NewStringUTF("Error: Prompt too long");
    }

    // Clear context
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Create batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // Decode prompt
    if (llama_decode(g_ctx, batch) != 0) {
      LOGe("Failed to decode prompt");
      return env->NewStringUTF("Error: Failed to decode");
    }

    // Generate tokens
    std::string result;
    int n_cur = tokens.size();
    int n_decode = 0;

    while (n_decode < max_tokens) {
      llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);

      // Check for end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        LOGi("Sampled EOG token at step %d", n_decode);
        if (n_decode == 0) {
          return env->NewStringUTF("[Debug: Sampled EOG immediately]");
        }
        break;
      }

      // Get token text
      char buf[256];
      int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        LOGe("Failed to convert token to piece");
        break;
      }

      std::string piece(buf, n);

      // Check for stop strings
      if (piece.find("<|eot_id|>") != std::string::npos ||
          piece.find("<|end_of_text|>") != std::string::npos ||
          result.find("User:") != std::string::npos) {
        break;
      }

      result += piece;

      // Prepare next batch
      batch = llama_batch_get_one(&new_token, 1);
      if (llama_decode(g_ctx, batch) != 0) {
        LOGe("Failed to decode");
        break;
      }

      n_cur++;
      n_decode++;
    }

    LOGi("Generated %d tokens", n_decode);
    if (result.empty()) {
      return env->NewStringUTF("[Debug: Generated 0 tokens]");
    }
    return env->NewStringUTF(result.c_str());
  } catch (const std::exception &e) {
    LOGe("Native generation exception: %s", e.what());
    std::string err = "Error: Native exception: ";
    err += e.what();
    return env->NewStringUTF(err.c_str());
  } catch (...) {
    LOGe("Unknown native exception");
    return env->NewStringUTF("Error: Unknown native exception");
  }
}

} // extern "C"
