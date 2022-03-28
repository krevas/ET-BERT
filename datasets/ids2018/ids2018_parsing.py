# sudo apt update
# sudo apt install tshark
# pip install pyshark

"""
    IDS2018 데이터 다운로드 : aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/Friday-23-02-2018/pcap.zip" "./"
    
    CSE-CIC-IDS2018 Fri-23-02-2018 데이터 기준
    
    공격 pcap 파일 : UCAP172.31.69.28
    
    - 공격 ip(18.218.115.60)와 공격 시간이 사전에 정해져 있고 시간을 기준으로 공격 유형이 다릅니다.
    
    - 위에 공격 pcap 파일을 제외한 pcap에는 모두 정상 데이터이므로 샘플링하여 사용하시면 됩니다.
    
    - pcap파일명에 있는 IP 주소를 ip.src로 설정하고 http로 필터링 해야 합니다.
    
    - 중복된 HTTP request 제거하고 사용하는 것을 추천합니다.
    
    ** 파일이 중간에서 잘린 pcap 파일이 간혹 있어서 에러가 발생할 때는 파이썬 라이브러리에서 pyshark/capture/capture.py 429~433줄을 주석 처리하시면 됩니다.
"""
import os
import re
import logging
from glob import glob

import pyshark
import nest_asyncio
import pandas as pd

nest_asyncio.apply()


def generate_attack_data(pcap_dir_path, output_file_path):
    attack_pcap = "UCAP172.31.69.28"
    display_filter = "ip.src == 18.218.115.60 and http"

    pcap = pyshark.FileCapture(
        os.path.join(pcap_dir_path, attack_pcap), display_filter=display_filter
    )

    writer = open(output_file_path, "w")
    writer.write("label\ttime\texpert_message\trequest_full_uri\taccept\n")

    line_cnt = 0
    try:
        for pkt in pcap:
            frame_time = pkt.frame_info.time.split(" ")[3][:8]
            if frame_time >= "23:04:00" or frame_time <= "00:05:00":
                writer.write(
                    f"Brute Force -Web\t{pkt.frame_info.time}\t{pkt.http.get('expert_message')}\t{pkt.http.get('request_full_uri')}\t{pkt.http.get('accept')}\n"
                )
            elif frame_time >= "02:00:00" and frame_time <= "03:15:00":
                writer.write(
                    f"Brute Force -XSS\t{pkt.frame_info.time}\t{pkt.http.get('expert_message')}\t{pkt.http.get('request_full_uri')}\t{pkt.http.get('accept')}\n"
                )
            elif frame_time >= "04:00:00" and frame_time <= "04:20:00":
                writer.write(
                    f"SQL Injection\t{pkt.frame_info.time}\t{pkt.http.get('expert_message')}\t{pkt.http.get('request_full_uri')}\t{pkt.http.get('accept')}\n"
                )
            line_cnt += 1
    except:
        pcap.close()
    writer.close()

    logger.info(f"Attack data size : {line_cnt}")


def generate_normal_data(pcap_dir_path, output_file_path):

    writer = open(output_file_path, "w")
    writer.write("label\ttime\texpert_message\trequest_full_uri\taccept\n")

    line_cnt = 0

    for pcap_file_path in sorted(glob(os.path.join(pcap_dir_path, "*"))):
        logger.info(f"Loading : {pcap_file_path}")

        source_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", pcap_file_path)[-1]
        display_filter = f"ip.src == {source_ip} and http"
        pcap = pyshark.FileCapture(pcap_file_path, display_filter=display_filter)

        try:
            for pkt in pcap:
                try:
                    if pkt.http.get("expert_message"):
                        writer.write(
                            f"Normal\t{pkt.frame_info.time}\t{pkt.http.get('expert_message')}\t{pkt.http.get('request_full_uri')}\t{pkt.http.get('accept')}\n"
                        )
                        line_cnt += 1
                except:
                    pass
        except:
            pcap.close()

    writer.close()

    logger.info(f"Normal data size : {line_cnt}")


def remove_duplicate(input_file_path, output_file_path):
    logger.info(f"Input path : {input_file_path}")
    logger.info(f"Output path : {output_file_path}")
    df = pd.read_csv(input_file_path, sep="\t")
    logger.info(f"Input data size : {len(df)}")
    df = df.remove_duplicate(["expert_message", "request_full_uri"])
    logger.info(f"Output data size : {len(df)}")
    df.to_csv(output_file_path, sep="\t", index=False)


if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)-10s] (line: %(lineno)d) %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_date = "20180223"

    generate_attack_data(
        pcap_dir_path=f"./pcap/{file_date}",
        output_file_path=f"./raw/{file_date}/attack.tsv",
    )

    remove_duplicate(
        input_file_path=f"./raw/{file_date}/attack.tsv",
        output_file_path=f"./raw/{file_date}/attack_no_dup.tsv",
    )

    generate_normal_data(
        pcap_dir_path=f"./pcap/{file_date}",
        output_file_path=f"./raw/{file_date}/normal.tsv",
    )

    remove_duplicate(
        input_file_path=f"./raw/{file_date}/normal.tsv",
        output_file_path=f"./raw/{file_date}/normal_no_dup.tsv",
    )
