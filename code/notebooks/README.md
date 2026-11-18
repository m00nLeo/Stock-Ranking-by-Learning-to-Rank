# notebooks/ — notebook chạy trên Google Colab

Thư mục này sẽ chứa các notebook orchestration (mount Google Drive, cài `requirements.txt`, gọi vào
package `vnrank/`). Notebook chỉ điều phối; logic chính nằm trong `vnrank/` để test được bằng `pytest`
trước khi chạy trên Colab.

Dự kiến tạo dần theo tiến độ charter mục 7 (chưa tạo, giữ chỗ ở bước 3.1):

| Notebook | Tạo ở bước | Nội dung |
| :--- | :--- | :--- |
| `00_setup_colab.ipynb` | S1 | Mount Drive, cài đặt, kiểm tra môi trường |
| `01_data_pipeline.ipynb` | 3.2-3.9 | Tải vnstock, làm sạch, universe, feature, chia dữ liệu |
| `02_trend_labels.ipynb` | 3.10-3.13 | Nhãn xu hướng, trạng thái thị trường, kiểm rò rỉ |
| `03_train_baseline_regression.ipynb` | 4.2-4.6 | B0-B4, R1-R3 |
| `04_train_ranking.ipynb` | 4.4-4.5 | L1 LambdaRank, gộp đa horizon |
| `05_train_trend_classifier.ipynb` | 4.10 | T2 |
| `06_test_one_shot.ipynb` | 4.7-4.9, 4.13-4.14 | Chạy test một lần, độ nhạy, DSR/PBO |
| `07_system_demo.ipynb` | S2-S12 | Giao diện Gradio, chatbot, bộ kiểm thử 40 câu |
