FAILURE_ENTITY = {
    "failure_mode": "Current trip during run",
    "failure_element": "Motor drive",
    "failure_effect": "",
    "root_cause": "Insufficient margin between SW limit and HW trip"
        }


STRUCTURE_INPUT = {
    "system": "Motor drive",
    "elements": [
        {
            "element": "Motor control",
            "failure_modes": [
                "Component break-down",
                "Unbalanced motor currents",
                "Incorrect interpretation zero-crossing",
                "No detection",
                "Welded relay",
                "Relay cannot close",
            ],
            "failure_causes": [
                "Cooling insufficient",
                "Compressor vibrations",
                "(Starting) Motor current too high for chosen components",
                "Overvoltage due to motor disconnect",
                "Under Voltage due to incorrect triggering ",
                "Live switching of relays",
                "Priority zero-crossing interrupt too low",
                "Open loop control",
                "No (correctly designed) snubber design ",
            ],
            "failure_effects": [
                "Motor cannot start",
                "Motor starts without soft start",
                "Overcurrent towards motor",
            ]
        },
        # {
        #     "element": "Control board",
        #     "failure_modes": [
        #         "False overcurrent detection"
        #     ],
        #     "failure_causes": [
        #         "ADC noise",
        #         "Incorrect threshold configuration"
        #     ],
        #     "failure_effects": []
        # }
    ]
}

