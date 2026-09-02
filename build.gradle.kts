plugins {
    id("com.android.application") version "9.3.0" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.3.21" apply false
}

buildscript {
    dependencies {
        // AGP 9 has built-in Kotlin; pin the KGP used by the Compose compiler plugin.
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.21")
    }
}
