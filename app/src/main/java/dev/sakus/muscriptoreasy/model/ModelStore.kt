package dev.sakus.muscriptoreasy.model

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File

class ModelStore(private val context: Context) {
    private val modelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun importPte(uri: Uri): File {
        val displayName = queryDisplayName(uri) ?: "muscriptor-model.pte"
        require(displayName.lowercase().endsWith(".pte")) {
            "ExecuTorch model must be a .pte file"
        }

        val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val output = File(modelDir, safeName)
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected model" }
            output.outputStream().use { out -> input.copyTo(out) }
        }
        return output
    }

    fun installedModels(): List<File> =
        modelDir.listFiles()
            ?.filter { it.isFile && it.extension.equals("pte", ignoreCase = true) }
            ?.sortedBy { it.name.lowercase() }
            .orEmpty()

    private fun queryDisplayName(uri: Uri): String? {
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) return cursor.getString(index)
        }
        return null
    }
}
