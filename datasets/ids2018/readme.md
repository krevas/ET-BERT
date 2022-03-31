# 데이터 설명

- 컬럼은 탭 문자로 구분되어 있습니다.

- 파일 구조는 label / time / expert_message / request_full_uri / accept / file_data 로 구성되어 있습니다.

- file_data는 request method가 POST인 경우에 함께 전송된 데이터를 뜻합니다.

- _no_dup 파일은 expert_message와 request_full_uri 기준으로 중복이 제거된 파일입니다.

- 날짜별 공격 유형은 다음과 같습니다.
	- Thur-15-02-2018 : DoS-GoldenEye
	- Fri-16-02-2018 : DoS-Hulk
	- Thur-22-02-2018 : Brute Force -Web / Brute Force -XSS / SQL Injection
	- Fri-23-02-2018 : Brute Force -Web / Brute Force -XSS / SQL Injection

- 데이터셋 homepage : https://www.unb.ca/cic/datasets/ids-2018.html


# 원본 pcap 파일 다운로드 방법

- 파일 목록 보기
	- aws s3 ls --no-sign-request "s3://cse-cic-ids2018" --recursive --human-readable --summarize

- 파일 받기
	- aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/Thursday-22-02-2018/pcap.zip" "./"