# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /sdk/tools/proguard/proguard-android.txt

# Keep llama.cpp JNI classes
-keep class de.kherud.llama.** { *; }
-keepclassmembers class de.kherud.llama.** { *; }
