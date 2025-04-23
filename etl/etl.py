# etl/etl.py

from app.modules.data_processing import update_all

# Only refresh API datasets once a day
update_all()   # api_only=True by default

print("✅ API pipelines updated.")
