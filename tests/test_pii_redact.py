import pytest
from verifier.pii_redactor import PIIRedactor


class TestPIIRedactor:
    def test_redact_simple_select(self):
        redactor = PIIRedactor()
        sql = "SELECT id, email, phone FROM users"
        safe_sql, cols = redactor.redact_sql(sql)
        assert "email" in cols
        assert "phone" in cols
        # Because sqlglot adds formatting, check lowercase presence
        lower_sql = safe_sql.lower()
        assert "md5(email)" in lower_sql
        assert "md5(phone)" in lower_sql
        assert "id" not in cols

    def test_redact_with_alias(self):
        redactor = PIIRedactor()
        sql = "SELECT u.email AS user_email FROM users u"
        safe_sql, cols = redactor.redact_sql(sql)
        assert "email" in cols
        lower_sql = safe_sql.lower()
        assert "md5(u.email)" in lower_sql
        assert "as user_email" in lower_sql

    def test_redact_no_pii(self):
        redactor = PIIRedactor()
        sql = "SELECT id, created_at, status FROM users"
        safe_sql, cols = redactor.redact_sql(sql)
        assert len(cols) == 0
        assert "md5" not in safe_sql.lower()

    def test_custom_redaction_method(self):
        redactor = PIIRedactor(redaction_method="mask")
        sql = "SELECT ssn FROM employees"
        safe_sql, cols = redactor.redact_sql(sql)
        assert "ssn" in cols
        assert "mask(ssn)" in safe_sql.lower()

    def test_regex_pattern_matching(self):
        redactor = PIIRedactor()
        sql = "SELECT pii_address, normal_field FROM data"
        safe_sql, cols = redactor.redact_sql(sql)
        assert "pii_address" in cols
        assert "md5(pii_address)" in safe_sql.lower()

    def test_already_hashed(self):
        redactor = PIIRedactor()
        # If it's already hashed, we should probably not double hash, but testing duckdb syntax
        sql = "SELECT md5(email) FROM users"
        safe_sql, cols = redactor.redact_sql(sql)
        # It shouldn't double wrap
        assert "md5(md5" not in safe_sql.lower()
