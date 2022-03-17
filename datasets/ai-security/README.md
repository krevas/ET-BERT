실행 예시
```python
python3 preprocess.py \
    --input_path ai_제공_데이터_clean.csv \
    --output_dir ./dataset \
    --min_count 20 \
    --split
```

* `--input_path`: 입력 파일 경로
* `--output_dir`: 출력 디렉토리 경로
* `--min_count`: 레이블 별 최소 데이터 수 (최소 데이터에 미달되는 레이블 데이터는 제거)
* `--split`: 출력 파일을 train / test로 split 할지 여부
