ARG paynt_base=randriu/paynt:latest

# Pull paynt
FROM $paynt_base

WORKDIR /opt/payntdev

COPY setup.bash setup.bash
COPY GILhack.patch GILhack.patch

RUN bash setup.bash
