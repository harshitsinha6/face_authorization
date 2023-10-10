
from authorization import Authorization
import util


video_source = 0
auth = Authorization(video_source)

auth.start_stream()

# util.get_snapshot_and_save()
