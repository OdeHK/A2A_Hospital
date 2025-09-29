import json
from typing import List, Dict

with open("updated_services.json", 'r', encoding='utf-8') as f:
    DATA = json.load(f)

def load_packages(data) -> Dict[int, Dict]:
    packages = data["packages"]
    # only take id, name, description, price
    packages = [{k: v for k, v in pkg.items() if k in ["id", "name", "description", "price", "services_ids"]} for pkg in packages]
    return packages

PACKAGES = load_packages(DATA)

def convert_packages_to_str(packages: List[Dict]) -> str:
    pkg_strs = []
    for pkg in packages:
        pkg_str = (
            f"ID: {pkg['id']}\n"
            f"Mô tả: {pkg['description']}\n"
            f"Giá: {pkg['price']:,} VND"
        )
        pkg_strs.append(pkg_str)
    return "\n\n".join(pkg_strs)

def get_package_by_id(data: Dict= DATA, ids: List[int]=[]) -> str:
    packages = PACKAGES
    services = data.get("services", [])

    selected = []
    for pkg in packages:
        if pkg["id"] in ids:
            # Lấy tên services đi kèm
            service_names = [services[sid-1]['name'] for sid in pkg.get("services_ids", [])]

            pkg_str = (
                f"Tên gói: {pkg['name']}\n"
                f"Mô tả: {pkg['description']}\n"
                f"Giá: {pkg['price']:,} VND\n"
                f"Dịch vụ đi kèm:\n  - " + "\n  - ".join(service_names) + "\n"
            )
            selected.append(pkg_str)

    return "\n\n".join(selected)