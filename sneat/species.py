class Species:
    def __init__(self, representative, callbacks):
        self.representative = representative
        self.members = [representative]
        self.id = callbacks['get_next_species_id']()
        self.stagnation = 0

    def add_member(self, genome):
        self.members.append(genome)