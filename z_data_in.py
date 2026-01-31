# importing required modules
import pandas as pd


def load_data():
    try:
        cargill_vessels = pd.read_csv(
            "data/_cargill_capsize_vessels.csv",
            parse_dates=["etd_date"],
            dayfirst=True,
        )

        cargill_cargoes = pd.read_csv(
            "data/_cargill_committed_cargoes.csv",
            parse_dates=["laycan_start_date", "laycan_end_date"],
            dayfirst=True,
        )

        market_vessels = pd.read_csv(
            "data/_market_vessels.csv",
            parse_dates=["etd_date"],
            dayfirst=True,
        )

        market_cargoes = pd.read_csv(
            "data/_market_cargoes.csv",
            parse_dates=["laycan_start_date", "laycan_end_date"],
            dayfirst=True,
        )
        port_distances = pd.read_csv(
            "data/_port_distances.csv"
        )


    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}") from e


    # Normalize dates
    # strip column names (important!)
    for df in (cargill_vessels, cargill_cargoes, market_vessels, market_cargoes):
        df.columns = df.columns.str.strip()

    # force datetime conversion (safe even if already datetime)
    cargill_cargoes["laycan_start_date"] = pd.to_datetime(
        cargill_cargoes["laycan_start_date"], errors="coerce"
    ).dt.date

    cargill_cargoes["laycan_end_date"] = pd.to_datetime(
        cargill_cargoes["laycan_end_date"], errors="coerce"
    ).dt.date

    market_cargoes["laycan_start_date"] = pd.to_datetime(
        market_cargoes["laycan_start_date"], errors="coerce"
    ).dt.date

    market_cargoes["laycan_end_date"] = pd.to_datetime(
        market_cargoes["laycan_end_date"], errors="coerce"
    ).dt.date

    cargill_vessels["etd_date"] = pd.to_datetime(
        cargill_vessels["etd_date"], errors="coerce"
    ).dt.date

    market_vessels["etd_date"] = pd.to_datetime(
        market_vessels["etd_date"], errors="coerce"
    ).dt.date

    # storing FFA report in dataframe
    ffa = pd.DataFrame([
        {"Route": "5TC", "Feb 26": 14157, "Mar 26": 18454, "Q4 25": 24336, "Q1 26": 16746, "Q2 26": 22436,
        "Q3 26": 25146, "Q4 26": 25418, "Q1 27": 16339, "Cal 26": 22437, "Cal 27": 21714, "Cal 28": 20289,
        "Cal 29": 19404, "Cal 30": 18943, "Cal 31": 18775, "Cal 32": 18682},
        {"Route": "C3 (Tubarao-Qingdao)", "Feb 26": 17833, "Mar 26": 20908, "Q4 25": 22819, "Q1 26": 19456,
        "Q2 26": 21475, "Q3 26": 23192, "Q4 26": 23592, "Q1 27": 19408, "Cal 26": 21929, "Cal 27": 20197,
        "Cal 28": 19988},
        {"Route": "C5 (West Australia-Qingdao)", "Feb 26": 6633, "Mar 26": 8717, "Q4 25": 9689, "Q1 26": 7700,
        "Q2 26": 9083, "Q3 26": 9288, "Q4 26": 9408, "Q1 27": 7392, "Cal 26": 8870},
        {"Route": "C7 (Bolivar-Rotterdam)", "Feb 26": 10625, "Mar 26": 11821, "Q4 25": 13157, "Q1 26": 11219,
        "Q2 26": 12210, "Q3 26": 12610, "Q4 26": 12986, "Q1 27": 11190, "Cal 26": 12256, "Cal 27": 11940,
        "Cal 28": 11540, "Cal 29": 11000, "Cal 30": 10900}
    ])

    ffa.columns = ffa.columns.str.replace(" ", "_", regex=False)

    # storing bunker forward curve in dataframe
    bunker = pd.DataFrame([
        # Location, Grade, Feb-26, Mar-26, Apr-26, May-26, Jun-26, Jul-26, Aug-26, Sep-26, Oct-26, Nov-26, Dec-26, Cal-27
        ["Singapore", "VLSFO", 491, 490, 489, 489, 487, 484, 482, 480, 479, 476, 474, 470],
        ["Singapore", "MGO",   654, 649, 642, 639, 637, 637, 637, 637, 637, 637, 637, 637],
        ["Fujairah",  "VLSFO", 479, 478, 477, 476, 475, 473, 471, 469, 467, 465, 463, 460],
        ["Fujairah",  "MGO",   640, 638, 636, 633, 632, 631, 630, 629, 628, 626, 624, 620],
        ["Durban",    "VLSFO", 436, 437, 437, 436, 434, 432, 430, 427, 423, 421, 419, 414],
        ["Durban",    "MGO",   511, 510, 509, 510, 507, 505, 502, 501, 499, 496, 493, 484],
        ["Rotterdam", "VLSFO", 468, 467, 466, 464, 463, 461, 460, 458, 456, 454, 453, 450],
        ["Rotterdam", "MGO",   615, 613, 610, 608, 606, 604, 603, 601, 600, 598, 597, 595],
        ["Gibraltar", "VLSFO", 475, 474, 473, 472, 470, 468, 466, 464, 462, 460, 458, 455],
        ["Gibraltar", "MGO",   625, 623, 621, 619, 617, 615, 614, 612, 610, 608, 606, 604],
        ["Port Louis","VLSFO", 455, 454, 454, 453, 451, 449, 448, 446, 444, 442, 440, 438],
        ["Port Louis","MGO",   585, 583, 581, 580, 579, 578, 577, 576, 575, 574, 573, 570],
        ["Qingdao",   "VLSFO", 648, 643, 639, 636, 633, 630, 628, 626, 624, 622, 620, 616],
        ["Qingdao",   "MGO",   838, 833, 828, 825, 823, 822, 821, 820, 818, 817, 816, 815],
        ["Shanghai",  "VLSFO", 650, 645, 638, 636, 633, 632, 630, 634, 633, 631, 630, 627],
        ["Shanghai",  "MGO",   841, 836, 829, 826, 824, 823, 822, 822, 820, 818, 816, 818],
        ["Richards Bay","VLSFO",442, 441, 441, 440, 438, 436, 434, 432, 430, 428, 426, 423],
        ["Richards Bay","MGO", 520, 519, 518, 516, 514, 512, 510, 508, 506, 504, 502, 500],
    ])

    bunker.columns = [
            "Location", "Fuel",
            "Feb_26", "Mar_26", "Apr_26", "May_26", "Jun_26", "Jul_26",
            "Aug_26", "Sep_26", "Oct_26", "Nov_26", "Dec_26", "Cal_27"
        ]
    

    return cargill_vessels, cargill_cargoes, market_vessels, market_cargoes, port_distances, ffa, bunker





if __name__ == "__main__":

    cargill_vessels, cargill_cargoes, market_vessels, market_cargoes, port_distances, ffa, bunker = load_data()

    print(cargill_vessels)
    print("\n\n\n\n")

    print(cargill_cargoes)
    print("\n\n\n\n")

    print(market_vessels)
    print("\n\n\n\n")

    print(market_cargoes)
    print("\n\n\n\n")

    print(port_distances)
    print("\n\n\n\n")

    print(ffa.to_string(index=False))
    print("\n\n\n\n")

    print(bunker)

    print("=" * 60)









