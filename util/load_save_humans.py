import json

def save_humans(frames, metadata):
    frames_list = []
    for frame in frames:
        bodies = []
        for human in frame:
            body = {}
            body['pairs'] = human.pairs
            body["uidx_list"] = list(human.uidx_list)
            body["score"] = human.score

            body_parts = {}
            for part_id, bp in human.body_parts.items():
                body_parts[bp.part_idx] = {
                    "x":bp.y,
                    "y":bp.x,
                    "score":bp.score,
                    "uidx":bp.uidx,
                }
            body["body_parts"] = body_parts

            bodies.append(body)
        frames_list.append({"bodies": bodies})
    
    # print(json.dumps({ "humans":people }, indent=4, sort_keys=True))
    with open("data/humans.json", 'w') as f:
        f.write(json.dumps({ "metadata":metadata, "frames":frames_list }, indent=4, sort_keys=True))
    