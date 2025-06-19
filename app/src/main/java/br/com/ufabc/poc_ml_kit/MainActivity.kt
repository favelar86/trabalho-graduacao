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
import android.widget.ImageView
import android.widget.PopupMenu
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.canhub.cropper.CropImageView
import com.google.android.material.button.MaterialButton
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var preprocessedImageView: ImageView
    private lateinit var inputImageBtn: MaterialButton
    private lateinit var recognizeTextBtn: MaterialButton
    private lateinit var cropImageView: CropImageView
    private lateinit var recognizeTextTv: TextView
    private var imageUri: Uri? = null
    private lateinit var progressDialog: AlertDialog

    private var isActivityActive = true
    private var hogDescriptor: HOGDescriptor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageBtn = findViewById(R.id.inputImageBtn)
        recognizeTextBtn = findViewById(R.id.recognizeTextBtn)
        cropImageView = findViewById(R.id.cropImageView)
        recognizeTextTv = findViewById(R.id.recognizeTextEt)
        preprocessedImageView = findViewById(R.id.preprocessedImageView)

        progressDialog = AlertDialog.Builder(this).apply {
            setView(ProgressBar(this@MainActivity).apply { isIndeterminate = true })
            setCancelable(false)
        }.create()

        inputImageBtn.setOnClickListener { showInputImageDialog() }
        recognizeTextBtn.setOnClickListener { recognizeTextFromCroppedImage() }

        cropImageView.setOnCropImageCompleteListener { _, result ->
            if (isActivityActive) {
                if (result.isSuccessful) {
                    result.bitmap?.let {
                        cropImageView.setImageBitmap(it)
                    } ?: showToast("Bitmap cortado é nulo")
                } else {
                    showToast("Corte falhou: ${result.error}")
                }
            }
        }

        if (OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Biblioteca OpenCV carregada com sucesso")
            initializeHogDescriptor()
        } else {
            Log.e("OpenCV", "Falha na inicialização do OpenCV")
            showToast("Falha ao carregar OpenCV")
        }
    }


    private fun recognizeTextFromCroppedImage() {
        recognizeTextTv.text = ""
        val bitmap = cropImageView.croppedImage
        if (bitmap == null) {
            showToast("Por favor, selecione e corte uma imagem primeiro")
            return
        }

        progressDialog.show()
        Log.d("Reconhecimento", "Tentando com ML Kit Text Recognition...")
        val inputImageMLKit = InputImage.fromBitmap(bitmap, 0)
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        recognizer.process(inputImageMLKit)
            .addOnSuccessListener { visionText ->
                if (!isActivityActive) return@addOnSuccessListener

                // Verifica se o ML Kit encontrou algum texto
                if (visionText.text.isNotBlank()) {
                    Log.d("MLKit_RESULT", "ML Kit encontrou texto: ${visionText.text}")
                    val digitsOnly = visionText.text.replace("[^0-9 ]".toRegex(), "")
                    val digitList = digitsOnly.trim().split("\\s+".toRegex())
                    val resultBuilder = StringBuilder()
                    digitList.forEachIndexed { index, digit ->
                        resultBuilder.append("var${index + 1} = $digit\n")
                    }

                    recognizeTextTv.text = resultBuilder.toString()
                    progressDialog.dismiss()
                } else {
                    // ML Kit funcionou mas não achou texto, então tenta o modelo customizado
                    Log.w("MLKit_RESULT", "Nenhum texto reconhecido pelo ML Kit, tentando com TensorFlow (HOG+MLP).")
                    processWithTensorFlowModel(bitmap)
                    progressDialog.dismiss()
                }
            }
            .addOnFailureListener { e ->
                if (!isActivityActive) return@addOnFailureListener

                // ML Kit falhou, então tenta o modelo customizado
                Log.e("MLKit_Error", "ML Kit falhou: ${e.message}", e)
                showToast("ML Kit falhou, tentando modelo TensorFlow")
                processWithTensorFlowModel(bitmap)
                progressDialog.dismiss()
            }
    }


    private fun processWithTensorFlowModel(bitmap: Bitmap) {
        if (hogDescriptor == null) {
            showToast("HOGDescriptor não inicializado!")
            return
        }
        try {
            loadTensorFlowModel().use { interpreter ->
                Log.d("TensorFlow", "Iniciando pré-processamento (Cinza -> Redimensionar -> Binarizar)")
                val processedMat = preprocessImage(bitmap)

                val processedBitmap = Bitmap.createBitmap(processedMat.cols(), processedMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(processedMat, processedBitmap)
                preprocessedImageView.setImageBitmap(processedBitmap)

                val descriptors = MatOfFloat()
                hogDescriptor?.compute(processedMat, descriptors)
                val hogFeatures = descriptors.toArray()
                processedMat.release()

                if (hogFeatures == null || hogFeatures.size != 1296) {
                    showToast("Erro ao extrair HOG. Tamanho: ${hogFeatures?.size}")
                    Log.e("TensorFlow", "Tamanho inesperado do vetor HOG: ${hogFeatures?.size}")
                    return
                }

                val inputTensor = arrayOf(hogFeatures)
                val outputTensor = Array(1) { FloatArray(10) }
                interpreter.run(inputTensor, outputTensor)

                val results = processTensorFlowOutput(outputTensor)
                recognizeTextTv.text = results
            }
        } catch (e: Exception) {
            Log.e("TensorFlowError", "Erro com modelo TensorFlow: ${e.message}", e)
            showToast("Erro ao processar com TensorFlow: ${e.message}")
        }
    }


    private fun preprocessImage(bitmap: Bitmap): Mat {
        val originalMat = Mat()
        Utils.bitmapToMat(bitmap.copy(Bitmap.Config.ARGB_8888, true), originalMat)

        val grayMat = Mat()
        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

        val resizedMat = Mat()
        Imgproc.resize(grayMat, resizedMat, Size(56.0, 56.0))

        val binaryMat = Mat()
        Imgproc.threshold(resizedMat, binaryMat, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)

        // Libera a memória das matrizes intermediárias
        originalMat.release()
        grayMat.release()
        resizedMat.release()

        // Retorna a imagem binarizada corretamente, pronta para o HOG
        return binaryMat
    }


    private fun loadTensorFlowModel(): Interpreter {
        assets.openFd("seven_segment_model_mlp.tflite").use { assetFileDescriptor ->
            FileInputStream(assetFileDescriptor.fileDescriptor).use { fis ->
                val fileChannel = fis.channel
                val startOffset = assetFileDescriptor.startOffset
                val declaredLength = assetFileDescriptor.declaredLength
                val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
                return Interpreter(mappedByteBuffer)
            }
        }
    }


    private fun processTensorFlowOutput(outputTensor: Array<FloatArray>): String {
        val outputArray = outputTensor[0]
        var maxIndex = 0
        var maxValue = -1f
        outputArray.forEachIndexed { index, value ->
            if (value > maxValue) {
                maxValue = value
                maxIndex = index
            }
        }
        val confidence = "%.2f".format(maxValue * 100)
        val resultText = "Dígito (TF): $maxIndex\nConfiança: $confidence%"
        val allProbabilities = outputArray.mapIndexed { index, value -> " $index: ${"%.3f".format(value)}" }.joinToString(";")
        Log.d("TF_Output", "$resultText | Probs: {$allProbabilities}")
        return resultText
    }


    private fun initializeHogDescriptor() {
        if (hogDescriptor == null) {
            val winSize = Size(56.0, 56.0)
            val blockSize = Size(16.0, 16.0)
            val blockStride = Size(8.0, 8.0)
            val cellSize = Size(8.0, 8.0)
            val nbins = 9
            hogDescriptor = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            Log.d("HOG", "HOGDescriptor inicializado.")
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        isActivityActive = false
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
            showToast("Permissão da câmera negada")
        }
    }


    private fun openCamera() {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "Nova Foto")
            put(MediaStore.Images.Media.DESCRIPTION, "Da Câmera")
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


    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}