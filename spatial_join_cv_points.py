from datetime import datetime
import os

from shapely.strtree import STRtree


def main():
    fields_to_extract = ["Rt", "Rt_nohab_all"]
    base_vectors = {
        "1992": (
            "D:/jeronimodata/Spring/Inspring/   coastal_risk_tnc_esa1992_md5_96c41f.gpkg"
        ),
        "2020": (
            "D:/jeronimodata/Spring/Inspring/   coastal_risk_tnc_esa2020_md5_764c7d.gpkg"
        ),
    }

    target_vector_path = "coastal_risk_tnc_esa1992_2020.gpkg"

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    target_vector_path = f"%s_{timestamp}%s" % os.path.splitext(target_vector_path)

    # Create Rt_1992 and Rt_2020 from the respective Rts in the base_vectors
    # Create Rt_nohab_all_1992 and Rt_2020 from the respective Rt_nohab_all in the base_vectors
    # Create new field Service_1992: Rt_nohab_all_1992 - Rt_1992
    # Create new field Service_2020: Rt_nohab_all_2020 - Rt_2020
    # Create new field Rt_change: Rt_2020 - Rt_1992
    # Create new field Service_change: Service_2020 - Service_1992


if __name__ == "__main__":
    main()
