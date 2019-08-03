# APIs

Private Key / Public Key / email address

Server - private key and public key -- store (public key, private key)
Give user - hashed email addr and public key

## Access Control Service

1. Checking for a valid entity

    http://162.246.156.210:5005/blockchain/checkEntity

    Request: Requires just the address of the entity.

    ```
    {
        "address": "0x121212"
    }
    ```

    Response: Returns true of false as validity.

    ```
    {
        "data": true
    }
    ```

2. Adding a new entity

    http://162.246.156.210:5005/blockchain/registerEntity

    Request:

    ```
    {
        "address": "0x787878",
        "valid": "True",
        "expiry": 0,
        "hash": "0x7777",
        "permission": 0,
        "grantFlag": "True"
    }
    ```

    Response: Nothing if created successfully. Transaction Error for unsuccessful execution.

3. Adding new access

    http://162.246.156.210:5005/blockchain/registerAccess

    Request:

    ```
    {
        "src": "0x121212",
        "dst": "0x787878",
        "valid": "True",
        "expiry": 0,
        "hash": "DOT10",
        "resource": "Room-3",
        "includes": ["building1:Door-2", "building1:Room-3"]
    }
    ```

    Response: Nothing if created successfully. Transaction Error for unsuccessful execution.

4. Checking the list of accessible resources. --- This one needs some more work, it isn't completed.

    http://162.246.156.210:5005/blockchain/checkAccess

    Request:

    ```
    {
        "hash": "DOT10"
    }
    ```

    Response:

    ```
    {
        "From Address": "0x121212",
        "Resource Head": "Room-3",
        "To Address": "0x787878"
    }
    ```





## Building Control Service

TODO