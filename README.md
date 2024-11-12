# Assignment-2-Machine-Learning-Deep-Learning

## Deskripsi Proyek
Proyek ini berisi empat studi kasus yang menerapkan algoritma Machine Learning dan Deep Learning untuk klasifikasi, clustering, dan regresi pada dataset yang berbeda. Setiap tugas disusun dalam Jupyter Notebook menggunakan bahasa Python.

---

## Rincian Tugas

### 1. Assignment Chapter 2 - Case 1: Klasifikasi Nasabah (Exited)
**Tujuan**: Membuat model klasifikasi untuk memprediksi apakah seorang nasabah akan “Exited” (keluar) dari bank berdasarkan data nasabah yang tersedia.

**Dataset**: `SC_HW1_bank_data.csv`

**Tahapan**:
- **Data Preprocessing**: Menghapus kolom yang tidak relevan, melakukan One-Hot Encoding pada data kategorikal, dan menormalkan data menggunakan MinMaxScaler.
- **Pemodelan**:
  - Model 1: Random Forest Classifier
  - Model 2: Support Vector Classifier (SVC)
  - Model 3: Gradient Boosting Classifier
- **Hyperparameter Tuning**: Menggunakan Grid Search untuk mencari parameter optimal.
- **Evaluasi**:
  - Metode evaluasi: accuracy_score, classification_report, dan confusion_matrix
  - Membandingkan performa dari ketiga model untuk menentukan hasil terbaik.

**Kesimpulan**: Gradient Boosting Classifier memberikan akurasi terbaik dengan waktu pemrosesan yang lebih cepat dibanding model lain.

---

### 2. Assignment Chapter 2 - Case 2: Clustering Data Nasabah
**Tujuan**: Melakukan segmentasi atau clustering nasabah untuk tujuan pengelompokan dengan KMeans Clustering (unsupervised learning).

**Dataset**: `cluster_s1.csv`

**Tahapan**:
- **Data Preparation**: Menghilangkan kolom yang tidak diperlukan dari dataset.
- **Penentuan Jumlah Cluster Terbaik**: Menggunakan Silhouette Score untuk menentukan jumlah cluster (nilai k) yang optimal.
- **Pemodelan dengan KMeans**: Melatih model KMeans berdasarkan jumlah cluster terbaik.
- **Evaluasi dan Visualisasi**:
  - Menampilkan scatter plot untuk menunjukkan distribusi data dalam cluster yang terbentuk.

**Kesimpulan**: Jumlah cluster terbaik ditentukan berdasarkan Silhouette Score. Visualisasi scatter plot menunjukkan hasil segmentasi data.

---

### 3. Assignment Chapter 2 - Case 3: Prediksi Harga Rumah di California
**Tujuan**: Membangun model regresi menggunakan TensorFlow-Keras untuk memprediksi harga rumah di California berdasarkan dataset yang disediakan.

**Dataset**: California House Price (dari Scikit-Learn)

**Tahapan**:
- **Persiapan dan Pemisahan Data**:
  - Mengonversi data ke dalam bentuk DataFrame
  - Memisahkan data ke dalam train, validation, dan test set
  - Standarisasi dan normalisasi fitur
- **Membangun Model Neural Network**:
  - Menyusun dua hidden layer dengan 30 neuron dan fungsi aktivasi ReLU
  - Menggabungkan input ganda sebelum memasuki output layer
- **Training dan Evaluasi**:
  - Mengatur learning rate, jumlah epochs, dan batch size
  - Melatih model dan mengevaluasi loss untuk memastikan model tidak overfitting
- **Menyimpan Model**: Menyimpan model yang sudah dilatih dan melakukan prediksi pada beberapa sampel baru.

---

### 4. Assignment Chapter 2 - Case 4: Deteksi Transaksi Fraud
**Tujuan**: Membangun model klasifikasi dengan PyTorch untuk mendeteksi transaksi fraud berdasarkan dataset Credit Card Fraud 2023.

**Dataset**: `Credit Card Fraud 2023`

**Tahapan**:
- **Impor Dataset dengan GPU**:
  - Mengunduh dan membaca dataset menggunakan cuDF (Pandas versi GPU)
  - Menghapus kolom ID dan melakukan standarisasi di GPU
- **Pemisahan Data dan Konversi ke Tensor**:
  - Menentukan fitur X dan target Y
  - Membagi data menjadi train dan test set menggunakan GPU
  - Mengonversi data ke bentuk Tensor untuk DataLoader PyTorch
- **Membangun Model Neural Network**:
  - Membuat arsitektur Multilayer Perceptron dengan 4 hidden layers
  - Mengatur parameter seperti epochs, jumlah layer, dan learning rate
- **Training dan Evaluasi**:
  - Melatih model dan memastikan akurasi mencapai setidaknya 95%. Jika perlu, dilakukan fine-tuning.

---

## Cara Menjalankan Kode

1. **Buka Google Colab**.
2. **Unggah file notebook**:
   - `02_Kelompok_B_1.ipynb` untuk Case 1 dan Case 2.
   - `02_Kelompok_B_2.ipynb` untuk Case 3 dan Case 4.
3. Jalankan sel dalam notebook secara berurutan dan ikuti petunjuk yang terdapat di dalamnya.
