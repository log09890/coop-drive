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
Ô Traffic: camera nhìn từ trên xuống, mũi tên = vị trí sau 2 s, vòng đỏ = sắp gặp ego trong < 3 s (TTC). Vòng sáng quanh radar và dòng "policy brakes because of" = đầu vào chi phối quyết định ga/phanh (gradient của actor). Console: chỉ số cả phiên + dải 60 s sự kiện. Va chạm → tự phát lại ~10 s cuối; `/replay`, `/gif`, `/focus <ô>`, ↑/↓ lịch sử lệnh, Tab tự hoàn thành.
Action thứ hai: > 0 là ga, < 0 là phanh. Chưa có model khớp observation thì tài xế mẫu lái.

### Giao thông đô thị kiểu Việt Nam (`--vn`, nên dùng Town02)

Shortcut "Coop-Drive VN Traffic". `simulation/vn_traffic.py`:
- 80 xe máy, 8 xe đạp, 12 ô tô, 50 người đi bộ băng qua đường bất kỳ đâu; bám đuôi sát, tốc độ lệch nhau, một phần vượt đèn đỏ / lờ biển báo, xe máy đổi làn liên tục. Xe máy ở xa được dời về quanh ego để mật độ luôn dày.
- Đèn đỏ thật (không ép xanh cho ego): xe máy và ô tô không xếp hàng mà dồn kín các ô từ vạch dừng trở lại, lấn cả làn ngược chiều; ô tô dàn ngang.
- Pha bất ngờ (ngẫu nhiên ~5 s, hoặc `/spawn`): `cutin` tạt đầu, `encroach` lấn làn ngược chiều, `overtake` vượt ẩu rồi chèn, `suddenstop` phanh gấp, `jaywalk` người chạy qua đường, `swerve` đang bên trái cắt sang vỉa hè phải, `wrongway` đi ngược chiều sát lề, `pullout` từ vỉa hè lao ra, `zigzag` lạng lách, `flock` đàn xe máy vượt hai bên, `surround` xe máy vây ô tô. Ngoài ra còn `ped`, `slow`, `parked`.
- Xe do script lái tự giữ khoảng cách theo kích thước xe; xe nào lật/bay được đặt lại trên đường; hybrid physics cho xe ở xa (> 70 m).
- Train/thu dữ liệu trong môi trường này: `bc.py collect --town Town02 --vn`, `continuous_driver.py --town Town02 --vn-traffic`. Chạy khoảng 6–7 bước/giây với dashboard.

### Hạ tầng tự hành (`--infra`, giai đoạn 1)

Shortcut "Coop-Drive Infra" hoặc `dashboard.py --town Town02 --vn --infra`. `simulation/infrastructure.py`:
- **RSU** ở mỗi ngã tư (8 ở Town02): LiDAR trên cột 6 m (góc dưới −40°), tự tách nền tĩnh và gom cụm thành danh sách vật thể (vị trí, vận tốc, loại); sai số đo được ~0.5 m.
- **Kênh V2X**: trễ 100 ms ± 30, mất 10% gói, tầm 300 m, sai số GPS 1.5 m. Tin nhắn: **CAM** (xe hợp tác báo vị trí/vận tốc của mình), **CPM** (RSU báo vật thể nó thấy, 10 Hz), **SPaT** (đèn báo màu và số giây còn lại, tính theo vòng quay cả nhóm đèn + pha đỏ toàn bộ).
- **Observation** thêm 22 số: 4 vật thể V2X gần nhất (vị trí, vận tốc tương đối, ngoại suy theo độ trễ), đèn phía trước trên tuyến (khoảng cách, màu, giây còn lại), độ "cũ" của thông tin. Không bật `--infra` thì 22 số này bằng 0 (cùng kiến trúc model). Model cũ (134 đầu vào) cần thu dữ liệu + BC lại (`bc.py collect/train --infra`).
- Dashboard: LiDAR của RSU trong ô LiDAR (ô vuông vàng), vật thể biết qua V2X trong ô Traffic (ô vuông xanh), khối V2X trong ô Reward; `/v2x latency|loss|range|gps <giá trị>`, `/v2x on|off`.

### Hạ tầng tự hành, giai đoạn 2: edge server, DENM, CAM cho mọi xe

- **CAM cho mọi xe có trang bị V2X**: mặc định 100% ô tô, 0% xe máy (xe máy Việt Nam gần như không có thiết bị); `/v2x cars|bikes <%>`.
- Kênh V2X giao tin **cho từng nơi nhận** (ego và các RSU), mất gói tính riêng từng cặp; DENM là tin gửi riêng (unicast).
- **Edge server** ở mỗi RSU trong vòng 80 m quanh ego: gộp vật thể LiDAR của RSU với CAM nghe được (bỏ trùng), dự đoán xung đột phía trước ego (TTC < 3 s, cách < 1.5 m) → gửi **DENM**. Observation thêm 2 số (đang có cảnh báo, TTC) → 24 số hạ tầng.
- Tài xế mẫu: DENM báo xung đột < 1.5 s → dừng, < 3 s → giảm còn 6 km/h.
- Kịch bản `occluded` (`/spawn occluded`): xe tải đỗ lấn vỉa hè, xe máy chờ trước đầu xe tải, lao ngang làn khi ego còn 16 m.
- `occlusion_eval.py --config ego|v2v|v2i|both --trials 10` → `results/occlusion_<config>.json`; `--table` in bảng so sánh: % biết lúc xe máy bắt đầu lao ra, % biết trước khi nó vào làn, số giây cảnh báo, % va chạm, khoảng cách gần nhất.
- Dashboard: nhãn DENM màu đỏ, dấu X tại điểm xung đột dự đoán, vạch DENM trên dải thời gian.

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
