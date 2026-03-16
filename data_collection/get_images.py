import ee
import re

def safe_name(text):
    text = re.sub(r"\s+", "_", text)         # spaces → underscore
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    return text[:80]
# --- Authenticate/initialize ---
ee.Initialize(project="ntl-semester-project")  # if this errors, run: earthengine authenticate

# --- Inputs ---
START = "2023-08-01"
END   = "2023-09-01"

FOOTPRINTS_ASSET = "projects/ntl-semester-project/assets/city_tile_footprints_2000_ascii"  
DRIVE_FOLDER = "VNP46A2_tiles"                               

SCALE = 500  # meters
BAND  = "Gap_Filled_DNB_BRDF_Corrected_NTL"

# --- Load composite ---
col = (ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
       .filterDate(START, END))

def mask_q0(img):
    return (img.updateMask(img.select("Mandatory_Quality_Flag").eq(0))
              .select(BAND)
              .rename("ntl"))

ntl_good = col.map(mask_q0).median()

# --- Load footprints ---
footprints = ee.FeatureCollection(FOOTPRINTS_ASSET)

# Bring a small list client-side to create tasks 
feat_list = footprints.toList(footprints.size())
n = footprints.size().getInfo()

print("Footprints:", n)
print(footprints.first().toDictionary().getInfo())

tasks = []
for i in range(n):
    f = ee.Feature(feat_list.get(i))
    region = f.geometry()  # your rectangle polygon

    # Pick good unique names (fallback if properties missing)
    city = f.get("city").getInfo()
    city = safe_name(city)
    name = f"{city}_{i}"
    

    task = ee.batch.Export.image.toDrive(
        image=ntl_good,
        description=f"ntl_chip_{name}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"ntl_chip_{name}",
        region=region,
        scale=SCALE,
        maxPixels=1e13
    )
    task.start()
    tasks.append(task)

print(f"Started {len(tasks)} export tasks. Check progress in the EE Tasks UI.")