import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

URL = "http://localhost:8000/practice_stream"
# URL = "huiren.daotest.net/sseai/practice_stream"
test_data = {
    "id": "题目id",
    "question": "为什么要提升合作客户肾宝片和已进场的二线产品有货门店数量？",
    "answer": "上半年上半年业绩完成。上半年上半年业绩完成好的城市经理合作客户进场产品，门店有货率达90%以上返至上半年，合作客户已进场，产品有货率低，嗯是导致城市经理业绩没有完成的主要原因，要完成下半年的目标，不提升合作客户进场产品的有货没店，数量是不可能实现的。",
    "standardAnswer": "1、上半年业绩完成好的城市经理，合作客户进场产品门店有货率都在90%以上，反之，上半年合作客户已进场产品有货率极低，是导致城市经理业绩没有完成的主要原因。2、要完成下半年目标，不提升合作客户进场产品有货门店数量是不可能实现的。",
    "qaTag": [
        "了解企业在二次腾飞期获得业绩"
    ],
    "unit": "题目所属的单元"
}


def send_request(data):
    response = requests.post(URL, json=data)
    return response.status_code, response.text


def send_requests_concurrently(data, concurrent_count, total_requests):
    with ThreadPoolExecutor(max_workers=concurrent_count) as executor:
        futures = [executor.submit(send_request, data) for _ in range(total_requests)]

        # 遍历每一个完成的任务
        for future in as_completed(futures):
            try:
                status_code, response_text = future.result()
                # print(f"Response status code: {status_code}, time is {datetime.now()}")
                a = str(response_text).replace('\n', '').replace('data: ', '')
                # if '[END]' in a:
                #     pass
                # else:
                #     print(a)
                print(a)
            except Exception as exc:
                print(f"Request generated an exception: {exc}")


print(datetime.now())
send_requests_concurrently(test_data, concurrent_count=10, total_requests=20)
