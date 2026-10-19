# Coop-Drive: PPO + VAE (idreesshaikh) + cooperative LiDAR (CODD)

Xe ego vẫn lái bằng PPO trên latent VAE của camera semantic như repo gốc. Thêm vào:
mỗi episode spawn `COOP_VEHICLES` xe autopilot quanh ego, mỗi xe (và ego) gắn LiDAR theo
cấu hình CODD và "chia sẻ" scan + pose. Các scan được hợp nhất về khung toạ độ ego rồi nén
thành `COOP_SECTORS` khoảng cách vật cản gần nhất theo từng góc, nối vào navigation obs
(obs PPO: 100 -> 100 + COOP_SECTORS).

- `simulation/coop.py` — phép biến đổi pose UE4 (như `vis.getTransform` của CODD) + `sector_distances`, chỉ cần numpy, dùng được cả với file `.hdf5` của CODD.
- `simulation/sensors.py` — `LidarSensor` (thông số LiDAR lấy từ `genSnippet.py` của CODD).
- `simulation/settings.py` — `COOP_*`, `LIDAR_*`. `COOP_SECTORS = 0` = chạy y như repo gốc (dùng được model pretrained).
- Model coop lưu riêng ở `preTrained_models/ppo_coop/<Town>/` vì kích thước obs khác.

## Chạy (đã cài sẵn trên máy này)

- CARLA 0.9.10.1 + AdditionalMaps: `E:\CARLA_0.9.10.1\WindowsNoEditor` (egg 0.9.10 đã chép vào `carla/`).
- Python 3.7.9: `E:\Python37`, venv của dự án: `venv\` (torch 1.13.1+cu117).
- Khi bật coop, môi trường chạy **synchronous mode, 20 FPS** (Traffic Manager của 0.9.10 bị crash ở async). Tắt coop thì vẫn async như repo gốc.

1. Bật server: `E:\CARLA_0.9.10.1\WindowsNoEditor\CarlaUE4.exe -quality-level=Low -windowed -ResX=640 -ResY=480`
2. Kiểm tra phần hợp nhất LiDAR (không cần CARLA): `venv\Scripts\python test_coop.py`
3. Train: `venv\Scripts\python continuous_driver.py --exp-name ppo` (thêm `--town Town02` nếu muốn)
4. Chạy model đã train: `venv\Scripts\python continuous_driver.py --exp-name ppo --train False`
5. Model gốc của tác giả: đặt `COOP_SECTORS = 0` trong `simulation/settings.py` rồi chạy lệnh ở bước 4.

### Học bắt chước (nhanh hơn RL thuần rất nhiều)

- `venv\Scripts\python bc.py collect --steps 40000` — tài xế mẫu (pure pursuit bám waypoint tuyến đường) tự lái và ghi (observation, thao tác), khoảng 25 phút, lưu sau mỗi vòng vào `data/`.
- `venv\Scripts\python bc.py train` — dạy actor bắt chước (và critic), vài phút, ghi thành checkpoint mới nhất (thư mục `ppo_coop/<Town>` phải trống).
- `venv\Scripts\python continuous_driver.py --exp-name ppo --load-checkpoint True` — fine-tune PPO từ model BC.

### Dashboard

`venv\Scripts\python dashboard.py` (hoặc `--model đường/dẫn.pth`): 6 ô — xe chạy, LiDAR hợp tác kiểu CODD, cảm biến, xử lý (VAE + PPO), logic (reward và luật dừng), tình huống giao thông (bản đồ + nhãn tự nhận diện).
Phím: `1` người băng qua đường, `2` xe chạy chậm phía trước, `3` xe đỗ chắn đường, `R` episode mới, `Esc` thoát.
Lưu ý: agent chỉ có lái và ga (không có phanh) như repo gốc, nên chưa xử lý được kịch bản 2 và 3.

Dọn xe/sensor sót lại sau một lần crash: `venv\Scripts\python clean_world.py`. Cả chuỗi thu dữ liệu → BC → fine-tune: `sh pipeline.sh`.

Model coop lưu ở `preTrained_models/ppo_coop/<Town>/`, metadata lưu ở `checkpoints/PPO_coop/<Town>/`, log TensorBoard ở `runs/`.
Repo gốc train khoảng 2 triệu bước (mặc định `--total-timesteps`); với RTX 2060 việc này mất nhiều giờ đến vài ngày.

Trích dẫn CODD (Arnold et al., RA-L 2021) nếu dùng lại code/dataset của họ.
