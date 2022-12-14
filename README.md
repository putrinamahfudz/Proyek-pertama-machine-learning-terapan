# Laporan Proyek Pertama Kelas Machine Learning Terapan - Putri Nur Aini Mahfudz
---
## Domain Proyek

Domain untuk proyek *machine learning* ini adalah bidang kesehatan. Proyek yang dibangun adalah sistem prediksi diagnosis stroke.

## **Latar Belakang**

Stroke adalah kondisi ketika pasokan darah ke otak terganggu karena penyumbatan (stroke iskemik) atau pecahnya pembuluh darah (stroke hemoragik). Kondisi ini menyebabkan area tertentu pada otak tidak mendapat suplai oksigen dan nutrisi sehingga terjadi kematian sel-sel otak. Tanpa suplai oksigen dan nutrisi, sel-sel pada bagian otak yang terdampak bisa mati hanya dalam hitungan menit. Akibatnya, bagian tubuh yang dikendalikan oleh area otak tersebut tidak bisa berfungsi dengan baik. 

Penyakit stroke termasuk dalam penyebab kecacatan nomor satu dan penyebab kematian nomor tiga di dunia setelah penyakit jantung dan kanker. Termasuk di Indonesia, stroke memiliki nilai prevalensi yang cukup tinggi [1]. 

Berdasarkan riset kesehatan dasar yang dilakukan oleh Kementerian Kesehatan (Kemenkes) tahun 2013, prevalensi stroke dengan usia di atas 15 tahun sebanyak 7 orang per 1000 penduduk. Dan pada tahun 2018 ini bukan turun, malah meningkat menjadi 10,9 per 1000 penduduk. 

Dari pernyataan tersebut, menunjukkan bahwa ada peningkatan kasus stroke di Indonesia. Maka dari itu dibangunlah sebuah model *machine learning* untuk membantu memprediksi apakah seseorang berpotensi terdiagnosis stroke atau tidak. Dengan adanya model *machine learning* ini diharapkan dapat dilakukan deteksi dini gejala stroke agar yang berpotensi terdiagnosis bisa ditangani lebih awal sebelum terlambat.


## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
- Faktor apa saja yang berpengaruh terhadap prediksi diagnosis penyakit stroke? 
- Bagaimana cara membuat model untuk memprediksi penyakit stroke dengan menggunakan *machine learning*?
- Bagaimana cara memilih model *machine learning* dengan akurasi paling baik?

### Goals
Tujuan dari proyek ini adalah:
- Mengetahui variabel atau fitur yang berpengaruh terhadap prediksi diagnosis penyakit stroke
- Mengetahui cara membuat model *machine learning* untuk memprediksi apakah terdiagnosis stroke atau tidak
- Mengetahui perbandingan beberapa algoritma model sehingga ditemukan akurasi yang paling baik

### Solution Statements
Untuk mendapatkan model *machine learning* dengan akurasi terbaik, saya akan membuat 2 model berbeda untuk dibandingkan, di antaranya:
- *Random Forest*, adalah sebuah algoritma dalam model *machine learning* yang bekerja dengan membangun beberapa decision tree dan menggabungkannya demi mendapatkan prediksi yang lebih stabil dan akurat. 
- *KNN*, adalah sebuah algoritma dalam model *machine learning* yang mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga, sehingga data-data yang cenderung serupa akan dekat satu sama lain. 

## Data Understanding

Dataset yang digunakan pada proyek ini diambil dari https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset. 

Dataset ini terdiri sebanyak 5110 baris dan 12 kolom.

Penjelasan mengenai variabel yang ada pada dataset tersebut adalah sebagai berikut:
- id: merupakan parameter yang bernilai unik yang dimiliki setiap subjek.
- gender: merupakan fitur yang menyatakan jenis kelamin.
- age: merupakan fitur yang menyatakan usia.
- hypertension: merupakan fitur yang menyatakan apakah mengidap hipertensi. 0: tidak, 1: ya.
- heart_disease: merupakan fitur yang menyatakan apakah mengidap penyakit jantung. 0: tidak, 1: ya.
- ever_married: merupakan fitur yang menyatakan apakah sudah pernah menikah. 
- work_type: merupakan fitur yang menyatakan jenis pekerjaan.
- Residence_type: merupakan fitur yang menyatakan jenis tempat tinggal.
- avg_glucose_level: merupakan fitur yang menyatakan kadar gula darah rata-rata.
- bmi: merupakan fitur yang menyatakan kategori berat badan.
- smoking_status: merupakan fitur yang menyatakan kategori merokok.
- stroke: merupakan fitur target yang menyatakan apakah terdiagnosis stroke.

Berikut adalah tahapan yang diperlukan untuk memahami dataset sebelum dilakukan *pre-processing*
- meload dataset menjadi dataframe.
- melihat informasi jumlah baris dan kolom.
- melihat informasi kolom pada dataset.
- melihat hitungan rata-rata, dll pada dataset.
- mengecek jumlah data kosong pada setiap kolom.
- mengecek apakah ada data yang terduplikat.
- mengecek apakah data sudah seimbang atau belum.


Berikut adalah tahapan dalam visualisasi data untuk memahami data sebelum dilakukan *pre-processing*
- *Univariate Analysis*, dilakukan untuk melihat distribusi data setiap variabel. 
    - distribusi data gender
        
        ![gender](https://user-images.githubusercontent.com/99728385/192228395-94488f75-ad66-4dd3-9bc7-c849a1785627.PNG)

    - distribusi data ever_married
    
        ![ever_married](https://user-images.githubusercontent.com/99728385/192228873-6ac96941-848e-4544-8f07-fd2e26aaa974.PNG)
        
    - distribusi data work_type
    
        ![work_type](https://user-images.githubusercontent.com/99728385/192228947-00cb30ee-ec1c-4b8e-a031-1b4a4558f43a.PNG)
        
    - distribusi data Residence_type
    
        ![Recidence_type](https://user-images.githubusercontent.com/99728385/192229011-2697297f-3c49-49ba-bbe4-dde553e3ba2f.PNG)
    
    - distribusi data age

        ![age](https://user-images.githubusercontent.com/99728385/192229104-5c0bfec6-e730-43ce-9504-2f37dd0cd1a2.PNG)

    - distribusi data hypertension
    
        ![hypertension](https://user-images.githubusercontent.com/99728385/192229171-55a7e775-7edf-4b88-b5f2-da61fb2bf190.PNG)

    - distribusi data heart_disease

        ![heart_disease](https://user-images.githubusercontent.com/99728385/192229285-028a40af-444b-41f8-bd9d-d1f3f9c145e8.PNG)

    - distribusi data avg_glucose_level

        ![avg_glucose_level](https://user-images.githubusercontent.com/99728385/192229508-d4d5b4f3-c2cd-480d-8a52-fbe662c43dd2.PNG)

    - distribusi data bmi
    
        ![bmi](https://user-images.githubusercontent.com/99728385/192229586-3719d809-1a31-40ce-827f-09cb2d40bb99.PNG)

    - distribusi data stroke
    
        ![stroke](https://user-images.githubusercontent.com/99728385/192229667-49a6f35d-b307-41e2-97de-99fc211c42aa.PNG)


## Data Preparation

Berikut adalah tahapan yang dilakukan dalam proses data *preparation*:
- menghapus kolom yang tidak diperlukan. kolom atau variabel yang dihapus adalah id, karena tidak memiliki kepentingan untuk dimasukkan ke dalam model.
- menghapus kategori yang tidak diperlukan. kategori yang dihapus adalah unknown pada kolom smoking_status dan other pada kolom gender.
- penanganan data yang hilang atau *missing values*. dalam dataset ini, ada sebanyak 201 data kosong pada kolom bmi. maka diterapkan teknik melakukan imputasi atau nilai pengganti. nilai pengganti yang digunakan adalah nilai rata-rata *(mean)*.
- melakukan *upsample* agar data seimbang. dalam dataset ini, ditemukan bahwa data belum seimbang, maka dilakukan upsample agar data menjadi seimbang dan menghasilkan prediksi yang bagus.
- mendeteksi *outliers*. dalam proyek ini saya menggunakan InterQuartile Range untuk mendeteksi outliers.
- melakukan *one hot encoding*. ini dilakukan pada data kategorikal agar datanya berubah menjadi data numerikal.
- melakukan *feature detection*. teknik ini dilakukan untuk melihat fitur apa saja yang memiliki peranan penting untuk membuat model. dari sini ditemukan bahwa fitur Age, Average Glucose Level, dan BMI memiliki peranan yang sangat penting untuk digunakan dalam membuat model.
- membagi dataset, dan melakukan scaling dengan *MinMaxScaler*. teknik ini dilakukan untuk membuat numerical data pada dataset memiliki rentang nilai (scale) yang sama. 

## Modeling

Setelah melakukan data *preparation*, langkah selanjutnya yang dilakukan adalah membuat model *machine learning*. Pada proyek ini akan dibuat 2 model yaitu *Random Forest* dan *KNN*.

- *Random Forest*, dalam mengimplementasikan algoritma ini, saya menggunakan method *RandomForestClassifier* dari sklearn.ensemble dengan argumen n_estimators=30 dan max_features=3. dan dihasilkan akurasi test score sebesar 0,97 dan confusion matrix score sebesar 0,98. 

    ![note1](https://user-images.githubusercontent.com/99728385/192223120-4bf4b5f4-df8d-485e-b76b-6c583b1d3965.PNG)

  
  Kelebihan dari algoritma yang ini adalah dapat memperkiraan variabel apa yang penting dalam klasifikasi, sedangkan kekurangan dari algoritma ini yaitu memiliki kompleksitas yang tinggi.

- *KNN*, dalam mengimplementasikan algoritma ini, saya menggunakan method *KNeighborsClassifier* dari sklearn.neighbors dengan argumen n_neighbors=2. dan dihasilkan akurasi test score sebesar 0,94 dan confusion matrix score sebesar 0,97.

    ![note2](https://user-images.githubusercontent.com/99728385/192223188-ca6f93f2-cd73-4603-84e2-f28edef1bd85.PNG)

  
  Kelebihan dari algoritma yang ini adalah cukup efektif terhadap data yang besar, sedangkan kekurangan dari algoritma ini yaitu perlu menentukan nilai parameter K terlebih dahulu.

## Evaluation
Setelah membangun dua model *machine learning*, maka dapat dibandingkan akurasi prediksinya untuk mendapatkan model dengan kinerja yang terbaik.
Didapatkan metriks akurasi sebagai berikut, agar lebih mudah dimengerti maka dapat ditampilkan dengan visualisasi:

![perbandingan_akurasi](https://user-images.githubusercontent.com/99728385/192131658-473cfe7b-4cd5-4bfb-aafc-650774809218.PNG)

Dari diagram di atas dapat diketahui bahwa model dengan *Random Forest* memiliki akurasi yang lebih tinggi dibanding *KNeighbors*.

## Kesimpulan
- Variabel atau fitur pada dataset yang berpengaruh dan memiliki peranan penting untuk membuat model *machine learning* adalah variabel Age, Average Glucose Level, dan BMI.
- Setelah dilakukan perbandingan akurasi prediksi antara dua model yang sudah dilatih, dihasilkan fakta bahwa model yang dibangun dengan algoritma *Random Forest* memiliki akurasi yang lebih tinggi dibanding *KNeighbors*.


## Referensi
[1] 
A. Puspitawuri, E. Santoso, C. Dewi, "Diagnosis Tingkat Risiko Penyakit Stroke Menggunakan Metode K-Nearest Neighbor dan Na??ve Bayes", *Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer Universitas Brawijaya,* Vol.3, No.4, hlm.3319-3324, 2019.
