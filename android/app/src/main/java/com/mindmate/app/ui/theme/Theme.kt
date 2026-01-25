package com.mindmate.app.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkColorScheme = darkColorScheme(
    primary = MindMatePrimaryDark,
    secondary = MindMateSecondary,
    tertiary = Pink80,
    background = MindMateBackgroundDark,
    surface = MindMateSurfaceDark,
    onPrimary = MindMateBackgroundDark,
    onSecondary = MindMateBackgroundDark,
    onTertiary = MindMateBackgroundDark,
    onBackground = AssistantMessageTextDark,
    onSurface = AssistantMessageTextDark
)

private val LightColorScheme = lightColorScheme(
    primary = MindMatePrimary,
    secondary = MindMateSecondary,
    tertiary = Pink40,
    background = MindMateBackground,
    surface = MindMateSurface,
    onPrimary = UserMessageText,
    onSecondary = UserMessageText,
    onTertiary = UserMessageText,
    onBackground = AssistantMessageText,
    onSurface = AssistantMessageText
)

@Composable
fun MindMateTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false, // Disabled for consistent branding
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
