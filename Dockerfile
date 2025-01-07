ARG paynt_base=randriu/paynt:latest

# Pull paynt
FROM $paynt_base

WORKDIR /opt/payntdev

RUN bash setup.bash
