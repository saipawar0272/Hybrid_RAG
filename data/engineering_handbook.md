# Engineering Handbook

## Incident severity
SEV-1 incidents cause a major production outage, critical data integrity issue, or broad customer impact. SEV-1 incidents require immediate incident commander assignment and continuous stakeholder updates.

SEV-2 incidents cause significant degradation or partial loss of a critical service. The owning engineering team should investigate promptly and provide regular updates.

## Change management
Production changes require a documented change record, impact assessment, rollback plan, test evidence, and approval according to the change risk level. Emergency changes must be documented retrospectively as soon as practical.

## Deployment rollback
Before deployment, identify the last known good version and validate rollback steps. If health checks fail after release, stop further rollout, assess customer impact, and follow the approved rollback procedure.
