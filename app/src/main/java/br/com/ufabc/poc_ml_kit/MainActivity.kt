package br.com.ufabc.poc_ml_kit

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.EditText
import android.widget.PopupMenu
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.canhub.cropper.CropImageView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    private lateinit var inputImageBtn: MaterialButton
    private lateinit var recognizeTextBtn: MaterialButton
    private lateinit var cropImageView: CropImageView
    private lateinit var recognizeTextEt: EditText
    private var imageUri: Uri? = null
    private lateinit var progressDialog: AlertDialog

    // Controla o status da Activity para evitar manipulação indevida
    private var isActivityActive = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        // Inicializa os elementos da UI
        inputImageBtn = findViewById(R.id.inputImageBtn)
        recognizeTextBtn = findViewById(R.id.recognizeTextBtn)
        cropImageView = findViewById(R.id.cropImageView)
        recognizeTextEt = findViewById(R.id.recognizeTextEt)

        // Configura o progressDialog
        progressDialog = AlertDialog.Builder(this).apply {
            setView(ProgressBar(this@MainActivity).apply { isIndeterminate = true })
            setCancelable(false)
        }.create()

        // Configura os eventos dos botões
        inputImageBtn.setOnClickListener { showInputImageDialog() }
        recognizeTextBtn.setOnClickListener { recognizeText() }

        // Configura o CropImageView
        cropImageView.setOnCropImageCompleteListener { _, result ->
            if (isActivityActive) {
                if (result.isSuccessful) {
                    result.bitmap?.let {
                        cropImageView.setImageBitmap(it)
                        showToast("Image cropped successfully!")
                    } ?: showToast("Cropped bitmap is null")
                } else {
                    showToast("Crop failed: ${result.error}")
                }
            }
        }

        // Inicializa o OpenCV diretamente
        if (OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "OpenCV library loaded successfully")
        } else {
            Log.d("OpenCV", "OpenCV initialization failed")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isActivityActive = false

        // Libera o ProgressDialog
        if (::progressDialog.isInitialized && progressDialog.isShowing) {
            progressDialog.dismiss()
        }
    }

    private fun showInputImageDialog() {
        PopupMenu(this, inputImageBtn).apply {
            menuInflater.inflate(R.menu.input_image_menu, menu)
            setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    R.id.menuCamera -> requestCameraPermission()
                    R.id.menuGallery -> openGallery()
                }
                true
            }
            show()
        }
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            showToast("Camera permission denied")
        }
    }

    private fun openCamera() {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "New Picture")
            put(MediaStore.Images.Media.DESCRIPTION, "From Camera")
        }
        imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        imageUri?.let { takePicture.launch(it) }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success && imageUri != null) {
            cropImageView.setImageUriAsync(imageUri)
        }
    }

    private val pickImageFromGallery = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            imageUri = result.data?.data
            imageUri?.let { uri -> cropImageView.setImageUriAsync(uri) }
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        pickImageFromGallery.launch(intent)
    }

    private fun recognizeText() {
        recognizeTextEt.text.clear() // Limpa o campo de texto

        val bitmap = cropImageView.getCroppedImage()
        if (bitmap != null) {
            try {
                val processedBitmap = preprocessImage(bitmap)

                val inputImage = InputImage.fromBitmap(bitmap, 0)

                // Inicializa o reconhecedor de texto com opções padrão
                val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

                progressDialog.show()
                recognizer.process(inputImage)
                    .addOnSuccessListener { visionText ->
                        if (isActivityActive) {
                            progressDialog.dismiss()

                            // Divide o texto reconhecido em palavras com base nos espaços
                            val words = visionText.text.split("\\s+".toRegex()).filter { it.isNotBlank() }
                            val results = words.mapIndexed { index, word -> "var${index + 1} = $word" }

                            // Exibe as variáveis concatenadas no campo de texto
                            recognizeTextEt.setText(results.joinToString("\n"))

                            // Log para depuração (opcional)
                            results.forEach { Log.d("OCR_RESULT", it) }
                        }
                    }
                    .addOnFailureListener { e ->
                        if (isActivityActive) {
                            progressDialog.dismiss()
                            showToast("Failed to recognize text: ${e.message}")
                        }
                    }
            } catch (e: Exception) {
                showToast("Error processing image: ${e.message}")
            }
        } else {
            showToast("Please select an image first")
        }
    }

    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        // Converter o Bitmap para Mat (formato OpenCV)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // 1. Converter para escala de cinza
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY)

        // 2. Aumentar o contraste da imagem (opcional, dependendo da qualidade da imagem)
        Imgproc.equalizeHist(mat, mat)

        // 3. Aplicar thresholding para garantir que o texto fique preto (ajustar o valor de threshold)
        Imgproc.threshold(mat, mat, 150.0, 255.0, Imgproc.THRESH_BINARY)

        // Converter de volta para Bitmap para ser usado no ML Kit
        val processedBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, processedBitmap)

        // Retornar a imagem pré-processada
        return processedBitmap
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
