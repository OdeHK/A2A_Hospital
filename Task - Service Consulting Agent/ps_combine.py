def package_to_text(data, services = True):
    texts = []
    if services:
        
        services_map = {s["id"]: s for s in data["services"]}

        for package in data["packages"]:
            total_price = 0
            package_name = package["name"]
            package_desc = package["description"]
            package_age = package["recommended_age"]
            service_texts = []

            for sid in package["services_id"]:
                if sid in services_map:
                    service = services_map[sid]
                    service_texts.append(f"{service['name']} ({service['price']} VND)")
                    total_price += service['price']
            
            # nối thành đoạn văn
            text = f"{package_name}: {package_desc}. Độ tuổi {package_age}. Bao gồm các dịch vụ: {', '.join(service_texts)}. Giá gói khám: {total_price}"
            texts.append(text)
            
    else:
        for package in data["packages"]:
            package_name = package["name"]
            package_desc = package["description"]

            text = f"{package_name}: {package_desc}. Bao gồm các dịch vụ: {', '.join(service_texts)}."
            texts.append(text)

    return texts