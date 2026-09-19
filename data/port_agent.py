
import pandas as pd


class PortOperationsAgent:

    # --------------------------------------------------------
    # HISTORICAL THRESHOLDS
    # Derived from 2025 AIS port operations data
    #
    # Median waiting ratio = 26.3%
    # 75th percentile      = 30.4%
    # --------------------------------------------------------

    NORMAL_WAITING_THRESHOLD = 0.263
    HIGH_WAITING_THRESHOLD = 0.304


    def __init__(
        self,
        dataset_path="port_operations_agent_dataset.csv"
    ):

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        self.df = pd.read_csv(
            dataset_path
        )


        # ----------------------------------------------------
        # VALIDATE REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "hour_key",
            "vessel_count",
            "waiting_vessel_count",
            "approaching_vessel_count",
            "departing_vessel_count",
            "average_speed",
            "port_throughput"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Dataset is missing required columns: "
                + str(missing_columns)
            )


        # ----------------------------------------------------
        # PREPARE TIMESTAMP
        # ----------------------------------------------------

        self.df["hour_key"] = pd.to_datetime(
            self.df["hour_key"],
            utc=True,
            errors="coerce"
        )

        self.df = (
            self.df
            .dropna(subset=["hour_key"])
            .sort_values("hour_key")
            .reset_index(drop=True)
        )


        if self.df.empty:

            raise ValueError(
                "Dataset contains no valid port observations."
            )


    # ========================================================
    # CALCULATE WAITING RATIO
    # ========================================================

    def calculate_waiting_ratio(
        self,
        waiting_vessels,
        vessel_count
    ):

        if vessel_count <= 0:
            return 0.0

        waiting_ratio = (
            waiting_vessels
            / vessel_count
        )

        return waiting_ratio


    # ========================================================
    # DETERMINE CURRENT PORT STATUS
    # ========================================================

    def determine_port_status(
        self,
        waiting_ratio,
        port_throughput
    ):

        # ----------------------------------------------------
        # CONGESTED
        #
        # Waiting ratio is at/above historical high threshold
        # AND
        # No Waiting -> Moored transitions occurred
        # during the previous 4 hours
        # ----------------------------------------------------

        if (
            waiting_ratio
            >= self.HIGH_WAITING_THRESHOLD

            and

            port_throughput == 0
        ):

            return "CONGESTED"


        # ----------------------------------------------------
        # ELEVATED
        #
        # Waiting ratio is at/above historical median
        # ----------------------------------------------------

        elif (
            waiting_ratio
            >= self.NORMAL_WAITING_THRESHOLD
        ):

            return "ELEVATED"


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        else:

            return "NORMAL"


    # ========================================================
    # GET CURRENT / HISTORICAL PORT STATUS
    # ========================================================

    def get_status(
        self,
        timestamp=None
    ):

        # ----------------------------------------------------
        # LATEST AVAILABLE HOUR
        # ----------------------------------------------------

        if timestamp is None:

            row = self.df.iloc[-1]


        # ----------------------------------------------------
        # REQUESTED HISTORICAL HOUR
        # ----------------------------------------------------

        else:

            timestamp = pd.to_datetime(
                timestamp,
                utc=True,
                errors="coerce"
            )

            if pd.isna(timestamp):

                return {
                    "error":
                        "Invalid timestamp"
                }


            timestamp = timestamp.floor("h")


            result = self.df[
                self.df["hour_key"]
                == timestamp
            ]


            if result.empty:

                return {

                    "error":
                        "No port data available "
                        "for this timestamp",

                    "requested_timestamp":
                        str(timestamp)
                }


            row = result.iloc[0]


        # ====================================================
        # RAW PORT OBSERVATIONS
        # ====================================================

        vessels = int(
            row["vessel_count"]
        )


        waiting_vessels = int(
            row["waiting_vessel_count"]
        )


        approaching_vessels = int(
            row["approaching_vessel_count"]
        )


        departing_vessels = int(
            row["departing_vessel_count"]
        )


        average_speed = float(
            row["average_speed"]
        )


        port_throughput = int(
            row["port_throughput"]
        )


        # ====================================================
        # AGENT CALCULATIONS
        # ====================================================

        # Calculate waiting ratio from raw vessel counts

        waiting_ratio = (
            self.calculate_waiting_ratio(
                waiting_vessels,
                vessels
            )
        )


        # Determine current operational port status

        port_status = (
            self.determine_port_status(
                waiting_ratio,
                port_throughput
            )
        )


        # ====================================================
        # AGENT OUTPUT
        # ====================================================

        return {

            "timestamp":
                str(row["hour_key"]),

            "vessels":
                vessels,

            "waiting_vessels":
                waiting_vessels,

            "approaching_vessels":
                approaching_vessels,

            "departing_vessels":
                departing_vessels,

            "average_speed_knots":
                round(
                    average_speed,
                    2
                ),

            # Calculated by agent
            "waiting_ratio":
                round(
                    waiting_ratio,
                    3
                ),

            "port_throughput_last_4h":
                port_throughput,

            # Determined by agent
            "port_status":
                port_status
        }
