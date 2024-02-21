#ifndef SQLPARSER_H
#define SQLPARSER_H

#include <string>
#include <vector>

enum class TokenType {
    SELECT,
    FROM,
    WHERE,
    COLUMN,
    LITERAL,
    OPERATOR,
    AND,
    OR,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;
};

class SQLParser {
public:
    static std::vector<Token> parse(const std::string& sql);
private:
    static TokenType getTokenType(const std::string& token);
};

#endif // SQLPARSE_H
