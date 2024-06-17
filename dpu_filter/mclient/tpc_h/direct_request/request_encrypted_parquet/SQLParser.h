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
    BETWEEN, // Added BETWEEN operator
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;
};

class SQLParser {
public:
    static std::vector<Token> parse(const std::string& sql);
    static std::string tokenTypeToString(TokenType type); // Added tokenTypeToString method declaration

private:
    static TokenType getTokenType(const std::string& token);
    static bool isColumnName(const std::string& token); // New helper method declaration
    static bool isLiteral(const std::string& token);    // New helper method declaration
};

#endif // SQLPARSER_H
