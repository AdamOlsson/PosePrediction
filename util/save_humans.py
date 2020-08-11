import json

def save_humans(humans, metadata):
    people = []
    for human in humans:
        body = {}
        body['pairs'] = human.pairs
        body["uidx_list"] = list(human.uidx_list)
        body["score"] = human.score

        bodyparts = {}
        for part_id, bp in human.body_parts.items():
            bodyparts[bp.part_idx] = {
                "x":bp.y,
                "y":bp.x,
                "score":bp.score,
                "uidx":bp.uidx,
                "limb":0
            }
        body["bodyparts"] = bodyparts

        people.append(body)
    
    # print(json.dumps({ "humans":people }, indent=4, sort_keys=True))
    with open("data/humans.json", 'w') as f:
        f.write(json.dumps({ "metadata":metadata, "humans":people }, indent=4, sort_keys=True))
    