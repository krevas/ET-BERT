#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import shutil


def fix_dataset(method):
    dataset_path = "F:\\dataset\\cstnet-tls1.3\\"

    for p, d, f in os.walk(dataset_path):
        for label in d:
            if label != "0_merge_datas":
                label_domain = label.split(".")[0]
                print(
                    f"I:\\mergecap.exe -w I:\\dataset\\{label_domain}.pcap I:\\{label}\\*.pcap"
                )
    return 0


def reverse_dir2file():
    path = "F:\\dataset\\"
    for p, d, f in os.walk(path):
        for file in f:
            shutil.move(os.path.join(p, file), path)
    return 0


def dataset_file2dir(file_path):
    for parent, dirs, files in os.walk(file_path):
        for file in files:
            label_name = file.split(".pcap")[0]
            os.mkdir(os.path.join(parent, label_name))
            shutil.move(os.path.join(parent, file), os.path.join(parent, label_name))
    return 0


def file_2_pcap(source_file, target_file):
    command = f"I:\\tshark.exe -F pcap -r {source_file} -w {target_file}"
    os.system(command)
    return 0


def clean_pcap(source_file):
    target_file = source_file.replace(".pcap", "_clean.pcap")
    clean_protocols = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp and frame.len > 80"'
    command = (
        f"I:\\tshark.exe -F pcap -r {source_file} -Y {clean_protocols} -w {target_file}"
    )
    os.system(command)
    return 0


def statistic_dataset_sample_count(data_path):
    dataset_label = []
    dataset_lengths = []

    tls13_flag = 1

    temp = []
    for p, d, f in os.walk(data_path):
        if p == data_path:
            dataset_label.extend(d)
        elif f == []:

            if p.split("\\")[-1] not in dataset_label:
                continue

            file_num = 0
            for pp, dd, ff in os.walk(p):
                file_num += len(ff)
            dataset_lengths.extend([file_num])
            temp.append(p)
        else:
            if tls13_flag == 1:
                if p.split("\\")[-1] not in dataset_label:
                    continue

                file_num = 0
                for pp, dd, ff in os.walk(p):
                    file_num += len(ff)
                dataset_lengths.extend([file_num])
                temp.append(p)

    print(f"label samples: {dataset_lengths}")
    print(f"labels: {dataset_label}")
    return dataset_lengths, dataset_label


if __name__ == "__main__":
    fix_dataset(["method"])
