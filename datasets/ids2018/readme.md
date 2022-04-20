# 데이터 설명

- pcap/ : 원본 pcap 파일

- raw/ : pcap에서 http/https 패킷만 추출한 데이터

- 추출된 데이터의 컬럼은 탭 문자로 구분되어 있습니다.

- 파일 구조는 label / stream_index / src_ip / src_port / dst_ip / dst_port / tcp_seg_len / time / expert_message / request_full_uri / accept / file_data 로 구성되어 있습니다.

- stream_index가 같으면 동일한 TCP 세션입니다. (src_ip / src_port / dst_ip / dst_port가 모두 동일하면 같은 stream_index를 가짐)

- Brute Force 공격은 하나의 TCP 세션에서 많은 request를 보내고,  DoS 공격은 TCP 세션 당 하나의 request만 보내는 특징이 있습니다.

- DoS나 DDoS 공격은 하나의 서버에서 제공할 수 있는 세션 포트 수가 제한적이라는 점을 이용한 자원 고갈 형태의 공격을 하기때문에 위와 같은 특징을 보이는 것으로 생각됩니다. 

- 반면에 Brute Force 공격은 관리자 계정 권한을 탈취하는 것이 목적이므로 여러 TCP 세션을 호출할 필요가 없기 때문에 하나의 세션에서 여러 request를 보내는 것으로 보입니다.

- file_data는 request method가 POST인 경우에 함께 전송된 데이터를 뜻합니다.

- _no_dup 파일은 expert_message와 request_full_uri, file_data 기준으로 중복이 제거된 파일입니다.

- 날짜별 공격 유형은 다음과 같습니다.
	- Thursday-15-02-2018 : DoS-GoldenEye
	- Friday-16-02-2018 : DoS-Hulk
	- Thursday-22-02-2018 : Brute Force -Web / Brute Force -XSS / SQL Injection
	- Friday-23-02-2018 : Brute Force -Web / Brute Force -XSS / SQL Injection

- 데이터셋 homepage : https://www.unb.ca/cic/datasets/ids-2018.html


# 원본 pcap 파일 다운로드 방법

- 파일 목록 보기
	- aws s3 ls --no-sign-request "s3://cse-cic-ids2018" --recursive --human-readable --summarize

- 파일 받기
	- aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/Thursday-22-02-2018/pcap.zip" "./"
