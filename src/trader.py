class Trader:
    def __init__(
        self,
        initial_cash=1000.0,
        max_long_inventory=5,
        max_short_inventory=-5
    ):
        self.cash = initial_cash
        self.yes_inventory = 0
        self.max_long_inventory = max_long_inventory
        self.max_short_inventory = max_short_inventory

    def can_buy_yes(self, quantity=1):
        return self.yes_inventory + quantity <= self.max_long_inventory

    def can_sell_yes(self, quantity=1):
        return self.yes_inventory - quantity >= self.max_short_inventory

    def buy_yes(self, price, quantity=1):
        if not self.can_buy_yes(quantity):
            return False

        self.cash -= price * quantity
        self.yes_inventory += quantity
        return True

    def sell_yes(self, price, quantity=1):
        if not self.can_sell_yes(quantity):
            return False

        self.cash += price * quantity
        self.yes_inventory -= quantity
        return True

    def get_position_value(self, market_price):
        return self.yes_inventory * market_price

    def get_total_value(self, market_price):
        return self.cash + self.get_position_value(market_price)

    def summary(self, market_price):
        return {
            "Cash": round(self.cash, 2),
            "YES Inventory": self.yes_inventory,
            "Position Value": round(
                self.get_position_value(market_price),
                2
            ),
            "Total Value": round(
                self.get_total_value(market_price),
                2
            )
        }