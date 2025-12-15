import os
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

# --- KONFIGURASI ---
# Load Model (Pastikan best.pt ada di satu folder dengan app.py)
model = YOLO('best.pt')

# Folder untuk simpan gambar
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- KAMUS DATA DAUR ULANG ---
# Ini menghubungkan Class Name dari YOLO ke Info Daur Ulang
RECYCLE_INFO = {
    'PET': {
        'full_name': 'Polyethylene Terephthalate',
        'rate': 'Tinggi (Sangat Mudah)',
        'desc': 'Biasa digunakan untuk botol air mineral. Diterima di hampir semua bank sampah.',
        'color': '#4CAF50' # Hijau
    },
    'HDPE': {
        'full_name': 'High-Density Polyethylene',
        'rate': 'Tinggi (Mudah)',
        'desc': 'Botol sampo, deterjen, tutup botol. Serat plastiknya kuat dan mudah didaur ulang.',
        'color': '#4CAF50'
    },
    'PVC': {
        'full_name': 'Polyvinyl Chloride',
        'rate': 'Sangat Rendah (Sulit)',
        'desc': 'Pipa, kabel, mainan anak. Mengandung klorin, berbahaya jika dibakar & sulit didaur ulang.',
        'color': '#F44336' # Merah
    },
    'LDPE': {
        'full_name': 'Low-Density Polyethylene',
        'rate': 'Sedang (Terbatas)',
        'desc': 'Kantong kresek, plastik wrap. Bisa didaur ulang tapi sering menyangkut di mesin.',
        'color': '#FF9800' # Oranye
    },
    'PP': {
        'full_name': 'Polypropylene',
        'rate': 'Tinggi (Mudah)',
        'desc': 'Gelas plastik minuman, sedotan, wadah makanan. Salah satu plastik paling banyak dicari pengepul.',
        'color': '#4CAF50'
    },
    'PS': {
        'full_name': 'Polystyrene',
        'rate': 'Rendah (Sulit)',
        'desc': 'Styrofoam, sendok plastik murah. Seringkali tidak diterima bank sampah karena nilai ekonomis rendah.',
        'color': '#F44336'
    },
    # Tambahan jaga-jaga jika ada class lain
    'PETE': {'full_name': 'PET (Variation)', 'rate': 'Tinggi', 'desc': 'Sama dengan PET.', 'color': '#4CAF50'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # 2. Simpan file asli sementara
            filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 3. PROSES DENGAN AI (Inference)
            # conf=0.4 artinya hanya deteksi yang yakin di atas 40%
            results = model.predict(filepath, conf=0.4) 
            
            # 4. Ambil Gambar Hasil (Annotated)
            # YOLO mengembalikan array BGR, kita simpan jadi gambar baru
            result_img_bgr = results[0].plot() 
            result_filename = "pred_" + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img_bgr)

            # 5. Ekstrak Data Teks
            detected_items = []
            names = model.names # Daftar nama kelas (0: HDPE, 1: PET, dst)
            
            for r in results:
                for c in r.boxes.cls:
                    class_name = names[int(c)]
                    # Cari info di kamus RECYCLE_INFO, kalau gak ada kasih default
                    info = RECYCLE_INFO.get(class_name, {
                        'full_name': 'Unknown Plastic', 
                        'rate': 'Unknown', 
                        'desc': 'Jenis plastik tidak dikenali.',
                        'color': 'gray'
                    })
                    
                    # Cek agar tidak duplikat di list (misal ada 2 botol PET)
                    if info not in detected_items:
                        detected_items.append({
                            'short_name': class_name,
                            **info
                        })

            # 6. Kirim data ke HTML result
            return render_template('result.html', 
                                   original_img=filename, 
                                   result_img=result_filename,
                                   items=detected_items)

    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)