class PersonTracker:

    def __init__(self):
        self.next_id = 1
        self.people = {}

    def update(self, detections):

        # Basic implementation.
        # Can later be replaced by
        # ByteTrack / DeepSORT.

        tracked = []

        for detection in detections:

            tracked.append({
                "id": self.next_id,
                **detection
            })

            self.next_id += 1

        return tracked