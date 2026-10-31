# EPL 2026-27 title forecast

Dự báo xác suất vô địch / top 4 / xuống hạng Premier League 2026-27, cập nhật tự động mỗi thứ Hai.

## Cách hoạt động

| Bước | File | Làm gì |
|---|---|---|
| Ingest | `src/epl_forecast/ingest.py` | Tải CSV kết quả từ [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (3 mùa gần nhất) |
| Model | `src/epl_forecast/model.py` | Dixon-Coles (Dixon & Coles, 1997, JRSS Series C 46(2)) có trọng số giảm dần theo thời gian |
| Simulate | `src/epl_forecast/simulate.py` | Monte Carlo 10.000 lần phần lịch còn lại; lịch còn lại = các cặp (chủ, khách) chưa đá vì EPL đá vòng tròn 2 lượt |
| Output | `outputs/` | `latest.csv`, `latest.md`, `ratings.csv`, `history.csv` (lịch sử theo tuần) |
| Schedule | `.github/workflows/weekly.yml` | Cron thứ Hai 06:00 UTC, test → chạy → commit kết quả |

## Chạy local

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m epl_forecast.run
```

## Giới hạn (biết trước)

- Chỉ dùng số bàn thắng, chưa có xG, chấn thương, chuyển nhượng.
- Tie-break: điểm → hiệu số → bàn thắng → ngẫu nhiên (bỏ qua đối đầu).
- `xi` (time decay) và `l2` đang là giá trị mặc định, cần tune bằng backtest trên các vòng đã đá.

## Việc tiếp theo

1. Backtest + calibration: Brier score / log-loss so với xác suất suy ra từ odds Bet365 có sẵn trong CSV.
2. Thêm xG (Understat) làm input.
3. Thí nghiệm DLinear/NLinear: dự báo chuỗi phong độ xG từng đội làm feature, so với EWMA.
4. Dashboard từ `history.csv` (đường xác suất vô địch theo tuần).
