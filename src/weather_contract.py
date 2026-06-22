class WeatherContract:
    def __init__(
        self,
        event_name,
        threshold,
        contract_type,
        city
    ):
        self.event_name = event_name
        self.threshold = threshold
        self.contract_type = contract_type
        self.city = city
        self.resolved = False
        self.outcome = None

    def describe(self):
        return (
            f"Contract: {self.event_name}\n"
            f"City: {self.city}\n"
            f"Type: {self.contract_type}\n"
            f"Threshold: {self.threshold}"
        )

    def resolve(self, realized_value):
        if self.contract_type == "rain_above":
            self.outcome = realized_value > self.threshold

        elif self.contract_type == "temperature_above":
            self.outcome = realized_value > self.threshold

        else:
            raise ValueError("Unknown contract type")

        self.resolved = True

        return self.outcome

    def payout(self, side):
        if not self.resolved:
            raise ValueError("Contract has not resolved yet")

        if side == "YES":
            return 1 if self.outcome else 0

        elif side == "NO":
            return 0 if self.outcome else 1

        else:
            raise ValueError("Side must be YES or NO")