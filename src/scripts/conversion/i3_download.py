#!/usr/bin/env python
import os
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobClient, BlobServiceClient

CONN_STR = os.environ['AZURE_STORAGE_CONNECTION_STRING']
GCD_CONTAINER_NAME = 'i3-gcd'


def enumerate_i3_files(number_of_files_to_download, i3_type):
    blob_names = []
    service_client = BlobServiceClient.from_connection_string(CONN_STR)
    container_client = service_client.get_container_client(i3_type)
    for item in container_client.walk_blobs():
        blob_list = container_client.list_blobs(
            item.name,
            prefix=i3_type
        )
        for blob in blob_list:
            blob_name = blob.name.split('/')[-1]
            blob_names.append(blob.name)
    if number_of_files_to_download > 0:
        blob_names = blob_names[0:number_of_files_to_download]
    return blob_names


def download_i3_files(blob_names, out_dir, i3_type):
    downloaded_files = []
    print('Downloading i3 blobs:')
    for i, blob_name in tqdm(enumerate(blob_names)):
        blob = BlobClient.from_connection_string(
            conn_str=CONN_STR,
            container_name=i3_type,
            blob_name=blob_name
        )
        out_file = out_dir.joinpath(blob_name.split('/')[-1])
        with open(out_file, 'wb') as save_file:
            blob_data = blob.download_blob()
            blob_data.readinto(save_file)
        downloaded_files.append(out_file)
    return downloaded_files


def download_i3_gcd_file(out_dir):
    container = ContainerClient.from_connection_string(
        conn_str=CONN_STR,
        container_name=GCD_CONTAINER_NAME
    )
    gcd_list = container.list_blobs()
    for blob in gcd_list:
        gcd_name = blob.name
    out_file = out_dir.joinpath(gcd_name)
    blob = BlobClient.from_connection_string(
        conn_str=CONN_STR,
        container_name=GCD_CONTAINER_NAME,
        blob_name=gcd_name
    )
    with open(str(out_file), 'wb') as blob_file:
        blob_data = blob.download_blob()
        blob_data.readinto(blob_file)
    return gcd_name
