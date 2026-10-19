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

`Dashboard.bat` (shortcut "Coop-Drive Dashboard") hoặc `venv\Scripts\python dashboard.py [--town Town02 --vn] [--model x.pth] [--expert]`.
6 ô: camera, LiDAR hợp tác kiểu CODD, cảm biến, VAE · PPO, reward, traffic (bản đồ + nhãn tình huống) và Console (log + lệnh).
Bấm `/` hoặc Enter để gõ lệnh, `/help` để xem: `/spawn <event>`, `/chaos on|off|<s>`, `/density <n>`, `/driver expert|ppo`,
`/weather clear|cloudy|rain|sunset`, `/pause`, `/reset`, `/snapshot`, `/clear`, `/quit`.
Action thứ hai: > 0 là ga, < 0 là phanh. Chưa có model khớp observation thì tài xế mẫu lái.

### Giao thông đô thị kiểu Việt Nam (`--vn`, nên dùng Town02)

Shortcut "Coop-Drive VN Traffic". `simulation/vn_traffic.py`:
- 80 xe máy, 8 xe đạp, 12 ô tô, 50 người đi bộ băng qua đường bất kỳ đâu; bám đuôi sát, tốc độ lệch nhau, một phần vượt đèn đỏ / lờ biển báo, xe máy đổi làn liên tục. Xe máy ở xa được dời về quanh ego để mật độ luôn dày.
- Đèn đỏ thật (không ép xanh cho ego): xe máy và ô tô không xếp hàng mà dồn kín các ô từ vạch dừng trở lại, lấn cả làn ngược chiều; ô tô dàn ngang.
- Pha bất ngờ (ngẫu nhiên ~5 s, hoặc `/spawn`): `cutin` tạt đầu, `encroach` lấn làn ngược chiều, `overtake` vượt ẩu rồi chèn, `suddenstop` phanh gấp, `jaywalk` người chạy qua đường, `swerve` đang bên trái cắt sang vỉa hè phải, `wrongway` đi ngược chiều sát lề, `pullout` từ vỉa hè lao ra, `zigzag` lạng lách, `flock` đàn xe máy vượt hai bên, `surround` xe máy vây ô tô. Ngoài ra còn `ped`, `slow`, `parked`.
- Xe do script lái tự giữ khoảng cách theo kích thước xe; xe nào lật/bay được đặt lại trên đường; hybrid physics cho xe ở xa (> 70 m).
- Train/thu dữ liệu trong môi trường này: `bc.py collect --town Town02 --vn`, `continuous_driver.py --town Town02 --vn-traffic`. Chạy khoảng 6–7 bước/giây với dashboard.

### Reward, observation và đánh giá (chế độ coop)

- Observation coop = (16 hướng + hành lang) **và độ thay đổi của chúng so với bước trước**, để agent thấy xe nào đang lao tới: 100 + 34 = 134 chiều.
- Reward = reward gốc − `safety_penalty`: suýt va chạm (vật cản trong làn < 3 m khi đang chạy), lách sát < 1 m khi đang chạy, đi vào đèn đỏ (chế độ VN), lái giật.
- Chế độ VN: tuyến ngẫu nhiên mỗi episode, chu kỳ đèn ngắn hơn (đỏ ~16 s).
- `share=False` / `--no-share`: chỉ dùng LiDAR của ego, để so sánh có/không có LiDAR hợp tác. `bc.py collect` ghi cả hai bản observation, `bc.py train --noshare` train model đối chứng.
- `evaluate.py`: seed cố định, policy lái tất định; ghi `results/<tên>.json` (tỉ lệ hoàn thành tuyến, va chạm/km, giây suýt va chạm/km, vượt đèn đỏ/km, tốc độ, độ giật).
- `sh pipeline_vn.sh`: thu dữ liệu VN → BC chia sẻ + chỉ ego → đánh giá → fine-tune PPO → đánh giá.

Dọn xe/sensor sót lại sau một lần crash: `venv\Scripts\python clean_world.py`. Cả chuỗi thu dữ liệu → BC → fine-tune: `sh pipeline.sh`.

Model coop lưu ở `preTrained_models/ppo_coop/<Town>/`, metadata lưu ở `checkpoints/PPO_coop/<Town>/`, log TensorBoard ở `runs/`.
Repo gốc train khoảng 2 triệu bước (mặc định `--total-timesteps`); với RTX 2060 việc này mất nhiều giờ đến vài ngày.

Trích dẫn CODD (Arnold et al., RA-L 2021) nếu dùng lại code/dataset của họ.
