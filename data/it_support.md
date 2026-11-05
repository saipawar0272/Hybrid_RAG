# IT Support Troubleshooting Handbook

## ERR-504 payment gateway timeout
ERR-504 indicates that the payment service did not receive a timely response from the upstream payment gateway. First, check the payment service health dashboard and confirm whether the gateway is experiencing an incident. Next, inspect request latency, timeout settings, and correlation IDs in the service logs. Do not retry a payment blindly: verify the transaction status first to avoid duplicate charges. Escalate unresolved incidents to the Payments on-call team with the correlation ID and timestamp.

## Password reset
Employees can reset their corporate password through the identity portal. If multi-factor authentication is unavailable, contact the IT service desk and complete identity verification. Never share passwords or one-time passcodes with support staff.

## VPN connection issue
Confirm internet connectivity, check whether the VPN client is up to date, and verify the employee's account is active. Capture the client error code and timestamp before escalating to Network Operations.

## Laptop performance
Check available disk space, memory pressure, startup applications, and pending operating system updates. Do not install unapproved cleanup utilities. Escalate persistent performance problems with diagnostic logs.
