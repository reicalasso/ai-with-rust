# 🚀 Rust ML v2.0 - Kurulum Rehberi

## 📋 Hızlı Başlangıç

### Yöntem 1: Otomatik Setup (Önerilen) ⚡

```bash
# Tek komut ile tüm setup
./setup.sh
./run_cuda.sh
```

### Yöntem 2: Manuel Setup 🔧

```bash
# 1. PyTorch yolunu öğren
python3 -c "import torch; print(torch.__path__[0])"

# 2. Environment dosyalarını oluştur
cp cuda_env.sh.example cuda_env.sh
cp run_cuda.sh.example run_cuda.sh

# 3. Yolları güncelle (editörde aç ve değiştir)
nano cuda_env.sh
nano run_cuda.sh

# 4. Derle ve çalıştır
source cuda_env.sh
cargo build --release
./run_cuda.sh
```

---

## 📁 Proje Dosyaları

### Environment Dosyaları

| Dosya | Açıklama | Git'e commit edilir mi? |
|-------|----------|------------------------|
| `cuda_env.sh.example` | Environment template | ✅ Evet |
| `run_cuda.sh.example` | Run script template | ✅ Evet |
| `cuda_env.sh` | Kişisel environment (auto-generated) | ❌ Hayır (.gitignore'da) |
| `run_cuda.sh` | Kişisel run script (auto-generated) | ❌ Hayır (.gitignore'da) |
| `setup.sh` | Otomatik setup script | ✅ Evet |

### Kaynak Kodlar

```
src/
├── main.rs              (302 satır) - Ana koordinasyon
├── models.rs            (147 satır) - ResNet, Attention, GAN
├── demos.rs             (235 satır) - Transfer Learning, RL, etc.
├── advanced.rs          (337 satır) - CNN, LSTM, VAE, Ensemble
├── showcases.rs         (309 satır) - Production showcases
└── advanced_features.rs (466 satır) - RAG, HITL, Quantization
```

---

## 🔧 Environment Yapılandırması

### PyTorch Yolu Bulma

```bash
# Yöntem 1: Python ile
python3 -c "import torch; print(torch.__path__[0])"

# Yöntem 2: pip show ile
pip show torch | grep Location

# Yöntem 3: find ile
find ~ -name "torch" -type d -path "*/site-packages/*" 2>/dev/null
```

### Yaygın Kurulum Yolları

#### 1. Local pip (en yaygın)
```bash
~/.local/lib/python3.13/site-packages/torch
```

#### 2. Conda environment
```bash
~/anaconda3/envs/ml/lib/python3.11/site-packages/torch
~/miniconda3/envs/ml/lib/python3.11/site-packages/torch
```

#### 3. System-wide
```bash
/usr/local/lib/python3.13/dist-packages/torch
/usr/lib/python3/dist-packages/torch
```

#### 4. Virtual environment
```bash
/path/to/venv/lib/python3.13/site-packages/torch
```

---

## 🔍 Sorun Giderme

### CUDA bulunamıyor

**Semptom:**
```
CUDA available: false
```

**Çözüm:**
```bash
# 1. PyTorch CUDA'nın çalıştığını kontrol et
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. Eğer Python'da CUDA çalışmıyorsa, PyTorch'u yeniden yükle
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Environment dosyalarını yeniden oluştur
./setup.sh
```

### Build hataları

**Semptom:**
```
error: linking with `cc` failed
```

**Çözüm:**
```bash
# 1. Environment'ı yükle
source cuda_env.sh

# 2. tch crate'i yeniden derle
cargo clean
cargo build --release

# 3. Hala hata varsa, LIBTORCH'u kontrol et
echo $LIBTORCH
ls -la $LIBTORCH/lib
```

### Library not found

**Semptom:**
```
error while loading shared libraries: libtorch_cuda.so
```

**Çözüm:**
```bash
# 1. Kütüphanenin varlığını kontrol et
ls -la ~/.local/lib/python3.13/site-packages/torch/lib/libtorch_cuda.so

# 2. LD_PRELOAD'u kontrol et
echo $LD_PRELOAD

# 3. run_cuda.sh'ı kullan (LD_PRELOAD'u otomatik ayarlar)
./run_cuda.sh
```

---

## 💡 İpuçları

### 1. Farklı Python Versiyonları

Sisteminizde birden fazla Python varsa:

```bash
# Python versiyonunu kontrol et
python3 --version

# PyTorch hangi Python'da kurulu?
python3.13 -c "import torch; print(torch.__path__[0])"
python3.11 -c "import torch; print(torch.__path__[0])"

# setup.sh'da Python versiyonunu belirle
python3.13 setup.sh  # örnek
```

### 2. Conda ile Kullanım

```bash
# Conda environment'ı aktif et
conda activate ml

# Setup çalıştır
./setup.sh

# Run
./run_cuda.sh
```

### 3. Hızlı Test

```bash
# Sadece CUDA test et
source cuda_env.sh
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Compile test
cargo check

# Quick run (debug mode, daha hızlı derler)
cargo run
```

### 4. Performance Optimization

```bash
# Release mode (optimize edilmiş, production)
cargo build --release
./run_cuda.sh

# Debug mode (hızlı derleme, development)
cargo build
./target/debug/rust-ml

# Verbose output
RUST_BACKTRACE=1 ./run_cuda.sh
```

---

## 🎯 Beklenen Çıktı

Başarılı bir çalıştırma şöyle görünmelidir:

```
╔════════════════════════════════════════════════════════════════╗
║        🦀 RUST ML v2.0 - PRODUCTION AI SHOWCASE 🦀             ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════╗
║     Device Information             ║
╚════════════════════════════════════╝
  CUDA available:  ✓
  cuDNN available: ✓
  CUDA devices:    1
  Active device:   GPU 0
```

### Performans Metrikleri

- **Total execution time:** ~2.2 saniye
- **Inference speed:** 13.5M+ samples/sec
- **Binary size:** 1.4 MB
- **13 AI/ML teknikleri:** Hepsi çalışıyor ✅

---

## 📚 Ek Kaynaklar

### Rust + ML
- [tch-rs documentation](https://github.com/LaurentMazare/tch-rs)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)

### CUDA
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)

### Rust
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)

---

## 🆘 Destek

Sorun mu yaşıyorsunuz?

1. README.md'yi okuyun
2. SETUP_GUIDE.md'yi (bu dosya) inceleyin
3. `./setup.sh` ile otomatik setup deneyin
4. Issue açın (GitHub)

---

## ✅ Checklist

Kurulumunuz tamamlandıysa:

- [ ] PyTorch CUDA çalışıyor (`python3 -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Environment dosyaları oluşturuldu (`cuda_env.sh`, `run_cuda.sh`)
- [ ] Proje derlendi (`cargo build --release`)
- [ ] Program çalıştı (`./run_cuda.sh`)
- [ ] CUDA aktif (çıktıda "CUDA available: ✓")
- [ ] Tüm 13 teknik başarılı (hepsi ✅)

**Hepsi tamam mı? Tebrikler! 🎉 Rust ML projesi hazır!**
