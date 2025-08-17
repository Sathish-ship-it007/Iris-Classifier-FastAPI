import requests

def main():
    url = "http://127.0.0.1:8000/predict"
    examples = [
        [5.1,3.5,1.4,0.2],
        [6.1,2.8,4.7,1.2],
        [6.3,3.3,6.0,2.5],
    ]
    for m in examples:
        r = requests.post(url, json={"measurements": m}, timeout=10)
        print(m, "->", r.json())

if __name__ == "__main__":
    main()