package dev.sakus.muscriptoreasy

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.sakus.muscriptoreasy.inference.LocalMuScriptorEngine
import dev.sakus.muscriptoreasy.model.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MuScriptorHome()
                }
            }
        }
    }
}

@Composable
private fun MuScriptorHome() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val store = remember { ModelStore(context.applicationContext) }
    val engine = remember { LocalMuScriptorEngine() }

    DisposableEffect(engine) {
        onDispose { engine.close() }
    }

    var audioUri by remember { mutableStateOf<Uri?>(null) }
    var audioName by remember { mutableStateOf<String?>(null) }
    var modelStatus by remember { mutableStateOf("No local model bundle loaded") }

    val audioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            audioUri = uri
            audioName = displayName(context, uri) ?: uri.lastPathSegment ?: "Selected audio"
        }
    }

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            modelStatus = "Importing model bundle…"
            scope.launch {
                val result = withContext(Dispatchers.IO) {
                    runCatching {
                        val bundle = store.importBundle(uri)
                        engine.load(bundle).getOrThrow()
                        bundle
                    }
                }
                modelStatus = result.fold(
                    onSuccess = {
                        "Loaded locally: ${it.variant} / ${it.dim}d " +
                            "(${formatBytes(it.totalBytes)}, context ${it.maxContext})"
                    },
                    onFailure = { "Model load failed: ${it.message}" },
                )
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 28.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("MuScriptor Easy", style = MaterialTheme.typography.headlineMedium)
        Text(
            "100% local Android transcription. The app intentionally has no INTERNET permission.",
            style = MaterialTheme.typography.bodyMedium,
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Audio", style = MaterialTheme.typography.titleMedium)
                Text(audioName ?: "No audio selected")
                Button(onClick = { audioPicker.launch(arrayOf("audio/*", "video/*")) }) {
                    Text("Choose audio / video")
                }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Local MuScriptor model", style = MaterialTheme.typography.titleMedium)
                Text(modelStatus)
                Button(
                    onClick = {
                        modelPicker.launch(
                            arrayOf("application/zip", "application/octet-stream", "*/*"),
                        )
                    },
                ) {
                    Text("Import .msa bundle")
                }
            }
        }

        Spacer(Modifier.height(4.dp))

        Button(
            modifier = Modifier.fillMaxWidth(),
            enabled = audioUri != null && engine.isLoaded && false,
            onClick = {},
        ) {
            Text("Transcribe")
        }
        Text(
            "ABI v1 model bundles are now loadable offline. Next milestone: exact Android STFT/log-mel and the greedy KV-cache generation loop.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private fun displayName(context: Context, uri: Uri): String? {
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

private fun formatBytes(bytes: Long): String {
    val mib = bytes / (1024.0 * 1024.0)
    return if (mib >= 1024.0) "%.2f GiB".format(mib / 1024.0) else "%.1f MiB".format(mib)
}
