class ForecastModel:
    def __init__(self, initial_probability):
        self.probability = initial_probability

    def get_yes_fair_value(self):
        return self.probability

    def get_no_fair_value(self):
        return 1 - self.probability

    def update_probability(self, new_probability):
        if new_probability < 0 or new_probability > 1:
            raise ValueError("Probability must be between 0 and 1")

        self.probability = new_probability

    def calculate_yes_edge(self, market_price):
        return self.get_yes_fair_value() - market_price

    def generate_signal(self, market_price, threshold=0.05):
        edge = self.calculate_yes_edge(market_price)

        if edge > threshold:
            return "BUY YES"

        elif edge < -threshold:
            return "SELL YES"

        else:
            return "NO TRADE"

    def calculate_position_size(
        self,
        market_price,
        threshold=0.05,
        max_quantity=3
    ):
        edge = abs(
            self.calculate_yes_edge(market_price)
        )

        if edge <= threshold:
            return 0

        scaled_edge = edge / threshold
        quantity = int(scaled_edge)

        return min(
            quantity,
            max_quantity
        )