"""
A module to connect with access control service
"""
import json
import typing as T
import requests

access_control_service_address = \
    'http://162.246.156.210:5005/blockchain/'


def check_valid_entity(identity: str) -> bool:
    """
    Check the address
    """
    response: requests.Response = requests.get(access_control_service_address+'checkEntity',
                                               json=json.dumps({'address': identity}))
    if response.status_code == 200:
        try:
            response_data: T.Dict[str, T.Any] = response.json()
            return response_data['data'] if response_data['data'] else False
        except Exception as e:
            print(e)
            return False
    else:
        return False


def add_new_entity(identity: str, hashed_identity: str) -> bool:
    """
    Add a new address
    Request format:
    {
        "address": "{identity}",
        "valid": "True",              // always
        "expiry": 0,                  // "%d.%m.%Y %H:%M:%S"
        "start": 0,                   // same as above
        // "hash": "{hashed_identity}",
        "permission": 0,              // 0 for read and 1 for write
        "grantFlag": "True"           // visitor: "False", organizer: "True"
    }
    """
    json_str = json.dumps({
        'address': identity,
        'valid': 'True',
        'expiry': 0,
        'hash': hashed_identity,
        'permission': 0,
        'grantFlag': 'True'
    })
    try:
        response: requests.Response = requests.post(
            access_control_service_address+'registerEntity', json=json_str)
        if response.status_code != 200:
            return False
    except Exception as e:
        print(e)
        return False
    return True


def add_new_access(src_identity: str, dst_identity: str,
                   hashed_identity: str,
                   resource: str, includes: T.List[str]) -> bool:
    """
    Add a new access to resource, which includes the 'includes' field
    Request format:
    {
        "src": "{src_identity}",          // organizer's email / identity
        "dst": "{dst_identity}",          // invitee's email / identity
        "valid": "True",                  // always
        "expiry": 0,                      // the one just set before
        "start": 0,                       // mentioned before
        "hash": "{hashed_identity}",      // key for access rule, unique all the time
        "resource": "{resource}",         // destination meeting room
        "imply": ["resources"]            // array: path along the destination, order matters
        "excludes": ["resources"]          // array: don't want someone to do something, order doesn't matter
    }
    """
    json_str = json.dumps({
        "src": src_identity,
        "dst": dst_identity,
        "valid": "True",
        "expiry": 0,
        "hash": hashed_identity,
        "resource": resource,
        "includes": includes
    })
    try:
        response: requests.Response = requests.post(
            access_control_service_address+'registerAccess', json=json_str)
        if response.status_code != 200:
            return False
    except Exception as e:
        print(e)
        return False
    return True


def check_accessible_resources():
    """
    TODO
    Request:
    {
        "src":
        "dst":
        "hash": "DOT10"
        "resource": "{resource}"
    }
    Response:
    {
        "valid": "",
        "main": "{destination room}"
    }
    // ask Kalvin for unspecified resources......
    // ask him whether the resource is in an accessible room......
    """
