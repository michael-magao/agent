import common.common as common
import re
import json
import sys
import os
from tqdm import tqdm

toc_client = common.get_toc_client(base_url="", username="", password="")
space_client = common.get_space_client(username="", password="")

# Search ip in TOC of lan type
def toc_lan(ip: str):
    query_with = {
        "with_platform": True,
        "with_state": True,
        "with_resource_nodes": True,
        "with_power_status": True,
        "with_basic_info": True,
        "with_hardware_info": True,
        "with_details": True,
        "with_tags": True,
    }
    filter = {
        "kind": "AND",
        "type": "binary_operator",
        "values": [
            {
                "value": ip,
                "method": "equals",
                "type": "atom",
                "key": "ip_lan",
            },
        ]
    }
    try:
        resp = toc_client.global_servers(ip, query_with, filter)
        if len(resp["data"]["servers"]) == 0:
            return False, "IP not found on TOC as an lan address"
        result = {
            "ip_lan": resp["data"]["servers"][0]["ip_lan"],
            "idc": resp["data"]["servers"][0]["idc"],
            "az": resp["data"]["servers"][0]["az"],
            "segment": resp["data"]["servers"][0]["segment"],
            "cpu_count": resp["data"]["servers"][0]["cpu_count"],
            "memory": resp["data"]["servers"][0]["memory"],
            "platform_name": resp["data"]["servers"][0]["platform"]["platform_name"],
        }
        return True, result
    except (StopAsyncIteration, StopIteration):
        return False, f"IP not found on TOC as an lan address: {ip}"
    except Exception as e:
        return False, f"An error occurred: {e}"

# Search ip in TOC of wan type
def toc_wan(ip: str):
    query_with = {
        "with_platform": True,
        "with_state": True,
        "with_resource_nodes": True,
        "with_power_status": True,
        "with_basic_info": True,
        "with_details": True,
        "with_tags": True,
    }
    filter = {
        "kind": "AND",
        "type": "binary_operator",
        "values": [
            {
                "value": ip,
                "method": "equals",
                "type": "atom",
                "key": "ip_wan",
            },
        ]
    }
    try:
        resp = toc_client.global_servers(ip, query_with, filter)
        if len(resp["data"]["servers"]) == 0:
            return False, "IP not found on TOC as an wan address"
        return True, resp
    except (StopAsyncIteration, StopIteration):
        return False, f"IP not found on TOC as an wan address: {ip}"
    except Exception as e:
        return False, f"An error occurred: {e}"

# Search ip in space container network.
def space_container_network(ip: str):
    try:
        response = space_client.list_ports_by_anchor(ip)
        if response["error"] != 0:
            return False, "Error when query space sdn api: {}".format(response["error"]["type"])
        if not response["data"].get("ports"):
            return False, "No ports found for ip: {}".format(ip)
        result = []
        for port in response["data"]["ports"]:
            ##  1Very complex result
            # result.append({
            #     "ip": port["ip"],
            #     "labels": {
            #         "cluster-id": port["labels"].get("cluster-id"),
            #         "endpoint-id": port["labels"].get("endpoint-id"),
            #         "pod-name": port["labels"].get("pod-name"),
            #         "node-name": port["labels"].get("node-name"),
            #         "owner": port["labels"].get("owner"),
            #         "pod": port["labels"].get("pod"),
            #     }
            # })

            # 2. Very plain pod name
            result.append(port.get("sdus")[0])
            print(port.get("sdus")[0])
        return True, result
    except Exception as e:
        return False, str(e)

# Search ip in space container network.
def space_container_network_with_sdu_to_service_conversion(ip: str):
    try:
        response = space_client.list_ports_by_anchor(ip)
        if response["error"] != 0:
            return False, "Error when query space sdn api: {}".format(response["error"]["type"])
        if not response["data"].get("ports"):
            return False, "No ports found for ip: {}".format(ip)
        result = []
        for port in response["data"]["ports"]:
            # 2. Very plain pod name
            result.append(port.get("sdus")[0])
            print(port.get("sdus")[0])
        return True, result
    except Exception as e:
        return False, str(e)

def is_valid_ip(ip: str) -> bool:
    ipv4_pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    ipv6_pattern = re.compile(r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$")
    return bool(ipv4_pattern.match(ip) or ipv6_pattern.match(ip))

def process_file(file_name: str):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    output_file = f"{file_name}"

if __name__ == "__main__":
    space_container_network_with_sdu_to_service_conversion("10.160.204.134")